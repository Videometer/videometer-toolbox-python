import json
import os
import pandas as pd
import sqlite3
from typing import List, Optional, Union, Tuple, Dict
from videometer.hips import ImageClass

class BlobDatabase:
    """
    A read-only interface for a Videometer Blob SQLite database.
    
    This class allows fetching blob images, querying IDs based on classifications,
    and generating Pandas DataFrames containing features and class labels.
    """

    def __init__(self, db_path: str):
        """
        Initialize the connection to the SQLite database and validate the version.

        Args:
            db_path (str): Path to the .blobdb SQLite file.

        Raises:
            FileNotFoundError: If the database file does not exist.
            ValueError: If the database version is not 4 or metadata is missing.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Connect in read-only mode using URI
        self.db_path = os.path.abspath(db_path)
        self.db_uri = f"file:{self.db_path}?mode=ro"
        
        # Perform initial version check immediately
        self._validate_version()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        return sqlite3.connect(self.db_uri, uri=True)

    def _validate_version(self):
        """
        Checks the 'metadata_t' table to ensure the database version is 6.
        """
        query = "SELECT value FROM metadata_t WHERE key = 'version'"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                
                if row is None:
                    raise ValueError("Invalid Database: 'version' key not found in metadata_t.")
                
                version_value = row[0]
                
                # Compare as string since the schema defines value as TEXT
                if version_value != '6':
                    raise ValueError(f"Unsupported database version: {version_value}. Expected version 6.")
                    
        except sqlite3.OperationalError as e:
            # Handle cases where metadata_t table might not exist at all
            raise ValueError(f"Invalid Database: Could not access metadata table. Original error: {e}")

    def get_blob(self, blob_id: str) -> ImageClass:
        """
        Extracts the blob image data for a specific blob ID and saves it to a temporary file.

        Args:
            blob_id (str): The unique UUID string of the blob.

        Returns:
            str: The file path to the temporary file containing the image bytes.
        
        Raises:
            ValueError: If the blob_id does not exist.
        """
        query = "SELECT blob_data FROM blobs_t WHERE blob_id = ?"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (blob_id,))
            row = cursor.fetchone()

            if row is None:
                raise ValueError(f"Blob with ID {blob_id} not found.")

            blob_bytes = row[0]

        return ImageClass.from_bytes(blob_bytes)

    def get_ids_by_reference_class(self, class_name: str) -> List[str]:
        """
        Retrieves blob IDs that are assigned to a specific Reference class.

        Args:
            class_name (str): The name of the reference class.

        Returns:
            List[str]: A list of blob UUIDs.
        """
        return self._get_ids_by_class_type(class_name, "reference")

    def get_ids_by_predicted_class(self, class_name: str) -> List[str]:
        """
        Retrieves blob IDs that are assigned to a specific Predicted class.

        Args:
            class_name (str): The name of the predicted class.

        Returns:
            List[str]: A list of blob UUIDs.
        """
        return self._get_ids_by_class_type(class_name, "prediction")

    def _get_ids_by_class_type(self, class_name: str, map_type: str) -> List[str]:
        """Helper method to query IDs based on label name and map type."""
        query = """
            SELECT b.blob_id
            FROM blobs_t b
            JOIN blob_labels_map m ON b.id = m.fk_blob_id
            JOIN labels_t l ON m.fk_label_id = l.id
            WHERE l.name = ? AND m.type = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (class_name, map_type))
            return [row[0] for row in cursor.fetchall()]

    def get_data_frame(self, ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generates a Pandas DataFrame containing blob features and classifications.

        The DataFrame includes:
        - Blob id
        - Reference Class (comma-separated if multiple)
        - Predicted Class (comma-separated if multiple)
        - All available features. Vector features are expanded into multiple columns 
          (e.g., Feature [0], Feature [1]).

        Args:
            ids (Optional[List[str]]): A list of blob UUIDs to include. 
                                       If None, all blobs are returned.

        Returns:
            pd.DataFrame: The constructed DataFrame.
        """
        with self._get_connection() as conn:
            # 1. Fetch Blobs
            blob_query = "SELECT id, blob_id FROM blobs_t"
            params = []
            
            if ids:
                placeholders = ','.join('?' for _ in ids)
                blob_query += f" WHERE blob_id IN ({placeholders})"
                params = ids

            blobs = pd.read_sql_query(blob_query, conn, params=params)
            
            if blobs.empty:
                return pd.DataFrame(columns=['Blob id', 'Reference Class', 'Predicted Class'])

            # Map internal integer ID to UUID for merging later
            blob_int_ids = tuple(blobs['id'].tolist())
            
            # Handle case where tuple has 1 element for SQL syntax "(x)" vs "(x,)"
            if len(blob_int_ids) == 1:
                ids_sql = f"({blob_int_ids[0]})"
            else:
                ids_sql = str(blob_int_ids)

            # 2. Fetch Labels (Reference and Predicted)
            # We fetch all mappings for these blobs
            labels_query = f"""
                SELECT m.fk_blob_id, m.type, l.name
                FROM blob_labels_map m
                JOIN labels_t l ON m.fk_label_id = l.id
                WHERE m.fk_blob_id IN {ids_sql}
            """
            labels_df = pd.read_sql_query(labels_query, conn)
            
            if not labels_df.empty:
                # Handle multiple labels per blob/type by aggregating them into a string
                # Group by blob_id and type, then join names with ", "
                labels_grouped = labels_df.groupby(['fk_blob_id', 'type'])['name'].apply(
                    lambda x: ', '.join(sorted(x))
                ).reset_index()

                # Pivot labels to columns
                labels_pivot = labels_grouped.pivot(index='fk_blob_id', columns='type', values='name')
                
                # Rename columns if they exist
                rename_map = {'reference': 'Reference Class', 'prediction': 'Predicted Class'}
                labels_pivot = labels_pivot.rename(columns=rename_map)
            else:
                labels_pivot = pd.DataFrame(columns=['Reference Class', 'Predicted Class'])

            # 3. Fetch Features
            features_query = f"""
                SELECT 
                    cf.fk_blob_id,
                    cf.value,
                    f.name as feature_name,
                    c.name as classifier_name
                FROM calc_features_t cf
                JOIN features_t f ON cf.fk_feature_id = f.id
                LEFT JOIN classifiers_t c ON f.fk_classifier_id = c.id
                WHERE cf.fk_blob_id IN {ids_sql}
            """
            features_raw = pd.read_sql_query(features_query, conn)

        # 4. Process Data
        
        # Combine Blob Info with Labels
        blobs.rename(columns={'blob_id': 'Blob id'}, inplace=True)
        blobs.set_index('id', inplace=True)
        
        main_df = blobs.join(labels_pivot, how='left')

        # Process Features
        if not features_raw.empty:
            expanded_rows = []

            # Pre-parse values to avoid repeated JSON parsing
            features_raw['parsed_value'] = features_raw['value'].apply(self._parse_feature_value)

            # Iterate over raw features to expand vectors
            # Note: Explicit iteration is used here to handle the dynamic column naming logic 
            # based on value types (list vs scalar) which is complex to vectorize cleanly.
            for _, row in features_raw.iterrows():
                b_id = row['fk_blob_id']
                val = row['parsed_value']
                f_name = row['feature_name']
                c_name = row['classifier_name']
                
                # Helper to format base name
                base_name = f"{f_name} ({c_name})" if c_name else f_name

                if isinstance(val, list):
                    # It's a vector feature -> Expand to multiple columns
                    for i, sub_val in enumerate(val):
                        # Format: Feature [0] (Classifier)
                        if c_name:
                            col_name = f"{f_name} [{i}] ({c_name})"
                        else:
                            col_name = f"{f_name} [{i}]"
                        
                        expanded_rows.append({
                            'fk_blob_id': b_id,
                            'col_name': col_name,
                            'final_value': sub_val
                        })
                else:
                    # It's a scalar (or len-1 list converted to scalar) -> Keep base name
                    expanded_rows.append({
                        'fk_blob_id': b_id,
                        'col_name': base_name,
                        'final_value': val
                    })

            if expanded_rows:
                # Create a long-format DataFrame from the expanded data
                features_expanded_df = pd.DataFrame(expanded_rows)

                # Deduplicate: In case of duplicates, keep the first one
                features_expanded_df.drop_duplicates(subset=['fk_blob_id', 'col_name'], keep='first', inplace=True)

                # Pivot to wide format
                features_pivot = features_expanded_df.pivot(
                    index='fk_blob_id', 
                    columns='col_name', 
                    values='final_value'
                )
                
                # Join features to main DataFrame
                main_df = main_df.join(features_pivot, how='left')

        # Reset index to drop the internal integer ID and return clean DF
        main_df.reset_index(drop=True, inplace=True)
        
        # Ensure 'Reference Class' and 'Predicted Class' exist even if no data found
        for col in ['Reference Class', 'Predicted Class']:
            if col not in main_df.columns:
                main_df[col] = None

        return main_df

    @staticmethod
    def _parse_feature_value(value: Union[float, str, int]) -> Union[float, list]:
        """
        Parses feature values according to specific rules:
        1. JSON arrays are 2D, discard outer array.
        2. If resulting array has 1 element, return as scalar.
        3. If value is already a scalar (REAL/float), return as is.
        """
        # If it's already a number (from REAL column), return it
        if isinstance(value, (int, float)):
            return value

        # If it's a string, try to parse as JSON
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                
                # Check for the specific "2D array" rule
                if isinstance(parsed, list):
                    if len(parsed) > 0 and isinstance(parsed[0], list):
                        # Discard outer array: [[1,2]] -> [1,2]
                        # Taking the first element of the outer array as the data
                        # (Assuming the "outer array" wraps the data row)
                        inner = parsed[0]
                    else:
                        inner = parsed

                    # Check for scalar reduction
                    if isinstance(inner, list) and len(inner) == 1:
                        return inner[0]
                    return inner
                
                return parsed
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, return original string
                return value
        
        return value

    def get_dataset(self, 
                    target_class_type: str = "reference", 
                    specific_classes: Optional[List[str]] = None, 
                    transform=None) -> "BlobDataset":
        """
        Factory method that creates a PyTorch Dataset context.

        Args:
            target_class_type (str): 'Reference' or 'Predicted'.
            specific_classes (List[str], optional): If provided, only includes blobs 
                                                    belonging to these class names.
            transform (callable, optional): Image transforms.

        Returns:
            BlobDataset: A dataset ready for DataLoader.
        """
        # 1. Query to get the IDs and Labels we want to include
        query = f"""
            SELECT b.id, l.name
            FROM blobs_t b
            JOIN blob_labels_map m ON b.id = m.fk_blob_id
            JOIN labels_t l ON m.fk_label_id = l.id
            WHERE m.type = ?
        """
        params = [target_class_type]

        if specific_classes:
            placeholders = ','.join('?' for _ in specific_classes)
            query += f" AND l.name IN ({placeholders})"
            params.extend(specific_classes)

        # Execute query on the MAIN thread/process
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            raise ValueError("No blobs found matching the criteria.")

        # 2. Build the Class <-> Index Mapping
        # Sort to ensure deterministic index assignment
        unique_classes = sorted(list(set(r[1] for r in rows)))
        class_to_idx = {name: i for i, name in enumerate(unique_classes)}
        idx_to_class = {i: name for name, i in class_to_idx.items()}

        # 3. Construct the list of samples (DB_ID, Label_Index)
        # This list is passed to the dataset, essentially "freezing" the view.
        samples = []
        for db_id, class_name in rows:
            samples.append((db_id, class_to_idx[class_name]))

        print(f"Created dataset with {len(samples)} samples across {len(unique_classes)} classes.")
        print(f"Classes: {class_to_idx}")

        return BlobDataset(
            db_path=self.db_path,
            samples=samples,
            class_map=idx_to_class,
            transform=transform
        )


class BlobDataset:
    """
    A lightweight view which can be used as a PyTorch Dataset created by BlobDatabase.
    It manages its own thread-safe SQLite connection for DataLoader workers.
    """
    def __init__(self, db_path: str, samples: List[Tuple[int, int]], 
                 class_map: Dict[int, str], transform=None):
        """
        Args:
            db_path (str): Path to the database file.
            samples (List): List of tuples (internal_db_id, label_index).
            class_map (Dict): Mapping of integer label_index -> class string name.
            transform (callable): PyTorch transforms.
        """
        self.db_path = db_path
        self.samples = samples
        self.class_map = class_map
        self.transform = transform
        
        # Connection is lazy-loaded per worker process
        self.conn = None
        # We need a URI to open in read-only mode
        self.db_uri = f"file:{os.path.abspath(db_path)}?mode=ro"

    def _get_connection(self) -> sqlite3.Connection:
        if self.conn is None:
            # This runs inside the worker process
            self.conn = sqlite3.connect(self.db_uri, uri=True)
        return self.conn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        db_id, label_idx = self.samples[idx]

        conn = self._get_connection()
        try:
            # We use the internal integer ID for fastest lookup
            cursor = conn.cursor()
            cursor.execute("SELECT blob_data FROM blobs_t WHERE id = ?", (db_id,))
            row = cursor.fetchone()
            
            if row is None:
                # Fallback or error handling if ID somehow missing
                raise ValueError(f"Blob ID {db_id} not found during iteration.")
                
            blob_bytes = row[0]
            
            image = ImageClass.from_bytes(blob_bytes).to_sRGB()

            if self.transform:
                image = self.transform(image)

            return image, label_idx

        except Exception as e:
            # Optional: Add robust error handling for corrupt images
            raise RuntimeError(f"Error loading sample {idx} (DB ID {db_id}): {e}")

    def __del__(self):
        """Ensure connection closes when dataset is destroyed."""
        if self.conn:
            self.conn.close()
