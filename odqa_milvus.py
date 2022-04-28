from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect()
TABLE_NAME = 'question_answering'
collection = None

#Deleting previouslny stored table for clean run
def create_mqa():
    if utility.has_collection(TABLE_NAME):
        collection = Collection(name=TABLE_NAME)
        collection.drop()

    field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True)
    field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",dim=768, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=TABLE_NAME, schema=schema)

    default_index = {"index_type": "IVF_FLAT", "metric_type": 'IP', "params": {"nlist": 200}}
    collection.create_index(field_name="embedding", index_params=default_index)

if utility.has_collection(TABLE_NAME):
    collection = Collection(name=TABLE_NAME)



search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

def find_similar(emb):
    collection.load()
    return collection.search(
	data=emb, 
	anns_field="embedding", 
	param=search_params, 
	limit=10, 
	expr=None,
	consistency_level="Strong"
)