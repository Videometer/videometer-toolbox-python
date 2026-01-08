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

