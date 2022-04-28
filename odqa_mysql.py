import pymysql
conn = pymysql.connect(host='localhost', user='aswin', port=3306, password='Mysql@123', database='ODQA',local_infile=True)
cursor = conn.cursor()
print("odqa")
# print("this is imp")
TABLE_NAME = 'context'

def create_context_table():
    #Deleting previouslny stored table for clean run
    drop_table = "DROP TABLE IF EXISTS " + TABLE_NAME + ";"
    cursor.execute(drop_table)
    try:
        sql = "CREATE TABLE if not exists " + TABLE_NAME + " (id TEXT, context TEXT);"
        sql = f"""
                CREATE TABLE if not exists {TABLE_NAME} (
                    id int(10) NOT NULL AUTO_INCREMENT,
                    context MEDIUMTEXT COLLATE utf8_bin NOT NULL, 
                    PRIMARY KEY (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
                AUTO_INCREMENT=1 ;"""
        cursor.execute(sql)
        print(f"{TABLE_NAME} table successfully!")
    except Exception as e:
        print("can't create a MySQL table: ", e)

def execute_query(query):
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        print("can't create a MySQL table: ", e)

def insert_context(context):
    """
    context should be array of contexts
    [con1, con2, ...]
    """
    # q = "select count(id) from context"
    # res = execute_query(q)
    # current_size = res[0][0]
    # next = current_size+1
    for text in context:
        sql = "INSERT INTO context (context) VALUES (%s)"
        cursor.execute(sql, (text))
        # next+=1 
    conn.commit()

def extract_context(id):
    q = f"select context from context where id = {id}"
    res = execute_query(q)
    return res[0]

def truncate_context():
    q = "truncate table context;"
    cursor.execute(q)

