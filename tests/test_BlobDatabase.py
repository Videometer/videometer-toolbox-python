import pytest
from videometer.BlobDatabase import BlobDatabase

def test_load_blob_features_as_dataframe():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # When all feature and class data is requested as a data frame
    df = db.get_data_frame()

    # Then a data frame is returned with feature and class data
    blob = df[df["Blob id"] == "5b2dc5aa-e52c-488b-8101-c4ce1075ae3a"].reset_index()
    assert blob["Reference Class"][0] == "Small"
    assert blob["Predicted Class"][0] == "Big"
    assert blob["Length (Unknown)"][0] == 25.4600983


def test_load_specific_blobs_as_dataframe():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # When feature and class data is requested for a subset of blobs
    ids = ["5b2dc5aa-e52c-488b-8101-c4ce1075ae3a"]

    df = db.get_data_frame(ids)

    # Then only those are returned
    assert df["Reference Class"][0] == "Small"
    assert df["Predicted Class"][0] == "Big"
    assert df["Length (Unknown)"][0] == 25.4600983

def test_load_specific_blob_image():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # When a blob's image data is requested
    id = "5b2dc5aa-e52c-488b-8101-c4ce1075ae3a"
    img = db.get_blob(id)

    # Then an ImageClass object of that blob is returned
    assert "NIR" in img.BandNames
    assert img.PixelValues.size

    # And its foreground mask is reconstructed from the blob image (not silently dropped)
    assert img.ForegroundPixels is not None
    assert img.ForegroundPixels.sum() > 0

def test_get_ids_by_reference_class():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # When blobs are filtered by reference class
    ids = db.get_ids_by_reference_class("Small")

    # Then the desired blob is returned
    assert '5b2dc5aa-e52c-488b-8101-c4ce1075ae3a' in ids

def test_get_ids_by_predicted_class():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # When blobs are filtered by predicted class
    ids = db.get_ids_by_predicted_class("Big")

    # Then the desired blobs are returned
    assert len(ids) == 3
    assert '5b2dc5aa-e52c-488b-8101-c4ce1075ae3a' in ids

def test_dataset():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    ds = db.get_dataset(specific_classes=["Small", "Large", "Double"])

    assert 3 == len(ds)
    assert 3 == len(ds[0][0].shape)

def test_dataset_get_blob_id():
    # Given a dataset built from a blob database
    db = BlobDatabase("TestData/3washers.blobdb")
    ds = db.get_dataset(specific_classes=["Small", "Large", "Double"])

    # Then each index maps to the UUID of the internal id stored in samples,
    # consistent with BlobDatabase.get_blob_id_for_db_id
    for i in range(len(ds)):
        db_id, _ = ds.samples[i]
        assert ds.get_blob_id(i) == db.get_blob_id_for_db_id(db_id)

    # And the known blob is among them
    blob_ids = [ds.get_blob_id(i) for i in range(len(ds))]
    assert "5b2dc5aa-e52c-488b-8101-c4ce1075ae3a" in blob_ids

def test_get_blob_id_for_db_id():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # And the internal integer id (blobs_t.id) of a known blob
    known_uuid = "5b2dc5aa-e52c-488b-8101-c4ce1075ae3a"
    with db._get_connection() as conn:
        db_id = conn.execute(
            "SELECT id FROM blobs_t WHERE blob_id = ?", (known_uuid,)
        ).fetchone()[0]

    # When the UUID is looked up from the internal id
    # Then the original UUID is returned
    assert db.get_blob_id_for_db_id(db_id) == known_uuid

def test_get_blob_id_for_db_id_not_found():
    # Given a blob database
    db = BlobDatabase("TestData/3washers.blobdb")

    # When a non-existent internal id is looked up
    # Then a ValueError is raised
    with pytest.raises(ValueError):
        db.get_blob_id_for_db_id(999999)

