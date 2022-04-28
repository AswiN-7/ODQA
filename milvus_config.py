from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect()
TABLE_NAME = 'question_answering'

#Deleting previouslny stored table for clean run
if utility.has_collection(TABLE_NAME):
    collection = Collection(name=TABLE_NAME)
    collection.drop()

field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True)
field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",dim=768, is_primary=False)
schema = CollectionSchema(fields=[field1, field2], description="collection description")
collection = Collection(name=TABLE_NAME, schema=schema)

default_index = {"index_type": "IVF_FLAT", "metric_type": 'IP', "params": {"nlist": 200}}
collection.create_index(field_name="embedding", index_params=default_index)


search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

