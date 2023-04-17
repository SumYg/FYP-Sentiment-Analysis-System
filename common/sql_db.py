import pymysql
from sshtunnel import SSHTunnelForwarder
# class PackedData:
# import emoji
# def remove_emoji(text):
#     return emoji.replace_emoji(text, replace='')

class MyDB:
    def __init__(self) -> None:
        with open('./cred/ssh.txt', 'r') as f:
            ssh_cred = f.read().split('\n')
        tunnel = SSHTunnelForwarder(('gatekeeper.cs.hku.hk', 22), ssh_password=ssh_cred[1], ssh_username=ssh_cred[0],
            remote_bind_address=("sophia.cs.hku.hk", 3306)) 
        tunnel.start()
        with open('./cred/sql.txt', 'r') as f:
            cred = f.read().split('\n')

        self.mydb = pymysql.connect(
            host="127.0.0.1",
            user=cred[0],
            password=cred[1],
            database=cred[0],
            charset='utf8mb4',
            port=tunnel.local_bind_port
        )
        self.order2id = {}

    def insert_keywords(self, keywords):
        self.insert_many('FYP_Keywords', ['date', 'keyword', 'positive_score', 'post_collected', 'display_order'], keywords)
        result = self.get_ids('FYP_Keywords', ['id', 'display_order'], f"date = '{keywords[0][0]}'")
        # print(result)
        for id, order in result:
            self.order2id[order] = id

    def insert_opinions(self, opinions, display_order, class_=0):
        """
        opinions: list of tuples (opinion, posts, likes, similar_opinions)
        """
        for opinion, posts, likes, agg_score, similar_opinions in opinions:
            # print(opinion, posts, likes, self.order2id[display_order])

            # opinion_id = self.insert('FYP_Opinion', ['class', 'text', 'posts', 'likes', 'agg_score', 'keyword_id'], [class_, remove_emoji(opinion), posts, likes, float(agg_score), self.order2id[display_order]])
            opinion_id = self.insert('FYP_Opinion', ['class', 'text', 'posts', 'likes', 'agg_score', 'keyword_id'], [class_, opinion, posts, likes, float(agg_score), self.order2id[display_order]])
            self.insert_similar_opinion(similar_opinions, opinion_id)

    def insert_similar_opinion(self, similar_opinions, opinion_id):
        # self.insert_many('FYP_Similar', ['opinion_id', 'similar_opinion', 'similarity'], [[opinion_id, remove_emoji(opinion), similarity] for opinion, similarity in similar_opinions])
        self.insert_many('FYP_Similar', ['opinion_id', 'similar_opinion', 'similarity'], [[opinion_id, opinion, similarity] for opinion, similarity in similar_opinions])


    def insert(self, table, columns, values):
        if values:
            sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['%s' for _ in range(len(values))])});"
            self.mycursor = self.mydb.cursor()
            # print(sql)
            # print(values)
            self.mycursor.execute(sql, values)
            print(self.mycursor.rowcount, "rows were inserted.")
            self.mydb.commit()

            self.mycursor.execute(f"SELECT LAST_INSERT_ID() FROM {table}")
            ids = self.mycursor.fetchall()

            # print the IDs of all the inserted rows
            print("IDs of all the inserted rows:", ids[0][0])

            self.mycursor.close()

            return ids[0][0]

    def insert_many(self, table, columns, values):
        if values:
            sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['%s' for _ in range(len(values[0]))])})"
            self.mycursor = self.mydb.cursor()
            # print(sql)
            # print(values)
            self.mycursor.executemany(sql, values)
            print(self.mycursor.rowcount, "rows were inserted.")
            self.mydb.commit()

            self.mycursor.close()

    def get_ids(self, table, columns, conditions):
        sql = f"SELECT {','.join(columns)} FROM {table} WHERE {conditions}"
        self.mycursor = self.mydb.cursor()
        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        self.mycursor.close()
        return result


if __name__ == '__main__':
    
    keywords = []

    db = MyDB()

    db.insert_keywords([('2023-03-19', ' 𝐚𝐯𝐚𝐢𝐥𝐚𝐛𝐥𝐞 𝐡𝐞𝐫𝐞 😀', '1', 1233, 1)])
    
    # db.insert_keywords([('2023-03-19', 'hello', '1', '2', 1233, 1), ('2023-03-19', 'world', '1', '2', 1233, 2)])
    # db.insert_opinions([('hello', 1, [('hi', 0.9), ('hello world', 0.8)]), ('world', 2, [])], 1)
    # db.insert_opinions([('this world', 1, [('hi this world', 0.9), ('hello world', 0.8)]), ('world!', 2, [])], 2)
    # db.insert_similar_opinion([('hi', 0.9), ('hello world', 0.8)], 2)

    # db.insert('FYP_Keywords', ['date', 'keyword', 'positive_score', 'negative_score', 'post_collected', 'display_order'], ['2023-04-18', 'hello', '1', '2', 1233, 1])
