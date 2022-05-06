from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect()
import odqa_mysql as odqa_mysql
import odqa_encoder as odqa_encoder

import json

TABLE_NAME = 'question_answering'
collection = None

#Deleting previouslny stored table for clean run
def create_mqa():
    global collection

    if utility.has_collection(TABLE_NAME):
        collection = Collection(name=TABLE_NAME)
        collection.drop()

    field1 = FieldSchema(name="ind", dtype=DataType.INT64, descrition="int64", is_primary=True)
    field2 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=False)
    field3 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",dim=768, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=TABLE_NAME, schema=schema)
    
    default_index = {"index_type": "IVF_FLAT", "metric_type": 'IP', "params": {"nlist": 200}}
    collection.create_index(field_name="embedding", index_params=default_index)

if utility.has_collection(TABLE_NAME):
    collection = Collection(name=TABLE_NAME)



search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

def find_similar(emb):
    global collection
    
    collection.load()
    return collection.search(
	data=emb, 
	anns_field="embedding", 
	param=search_params, 
	limit=10, 
	expr=None,
	consistency_level="Strong"
)



def push_context_to_milvus():
    global collection

    db_fp = r"database_handler.json"
    file = open(db_fp)
    database_handler = json.loads(file.read())
    file.close()

    start= database_handler['milvus_rows']
    end= start+database_handler["batch"]
    index = database_handler['milvus_rows']
    
    query = f"select * from context where id between {start} and {end} ;"
    res = odqa_mysql.execute_query(query)

    for id, context in res:
        emb = odqa_encoder.encode(context)
        collection.insert([[index], [id], emb])
        index+=1  
           
    database_handler['milvus_rows'] = end
    database_handler['milvus_index'] = index

    file = open(db_fp,"w")
    json.dump(database_handler, file)
    file.close()
    
    mysql_size = odqa_mysql.execute_query("select count(*) from QA_DATASET")[0][0]
    return f"mysql : {mysql_size}\nmilvus : {collection.num_entities}"