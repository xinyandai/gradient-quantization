import tests
from models import BaseModel


def test_create_base_model():
    model = BaseModel()
    assert model != None
