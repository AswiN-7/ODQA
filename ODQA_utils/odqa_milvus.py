from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
connections.connect()
import ODQA_utils.odqa_mysql as odqa_mysql
import ODQA_utils.odqa_encoder as odqa_encoder

import json

TABLE_NAME = 'question_answering'
collection = None

#Deleting previouslny stored table for clean run
def create_mqa():
    if utility.has_collection(TABLE_NAME):
        collection = Collection(name=TABLE_NAME)
        collection.drop()

    field1 = FieldSchema(name="ind", dtype=DataType.INT64, descrition="int64", is_primary=True)
    field2 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=False)
    field3 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector",dim=1024, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
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
	output_fields = ["id"],
	consistency_level="Strong"
)



def push_context_to_milvus():
    print("\n\n")
    db_fp = r"ODQA_utils\database_handler.json"
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
        indexs = []
        ids = []
        for i in range(len(emb)):
            indexs.append(index)
            index+=1  
            ids.append(id)
        # print(emb, indexs, ids)
        collection.insert([indexs, ids, emb])
           
    database_handler['milvus_rows'] = end
    database_handler['milvus_index'] = index

    file = open(db_fp,"w")
    json.dump(database_handler, file)
    file.close()
    
    mysql_size = odqa_mysql.execute_query("select count(*) from QA_DATASET")[0][0]
    return f"mysql : {mysql_size}\nmilvus : {collection.num_entities}"