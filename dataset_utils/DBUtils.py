import threading
import pymysql
from config import known_face

host = "127.0.0.1"
user = "root"
pwd = "123456"
port = 3306
database = "face_recogintion"


class MysqlDatabase(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(MysqlDatabase, "_instance"):
            with MysqlDatabase._instance_lock:
                if not hasattr(MysqlDatabase, "_instance"):
                    MysqlDatabase._instance = object.__new__(cls)
                    MysqlDatabase.conn = pymysql.connect(host=host, port=port, user=user, password=pwd, charset="utf8",
                                                         database=database, cursorclass=pymysql.cursors.DictCursor,
                                                         autocommit=1)
        return MysqlDatabase._instance

    def queryDataToDist(self, sql, printSql=True, printResult=True):
        cursor = self.conn.cursor()
        if printSql:
            print("执行sql---{}".format(sql))
        sql = str(sql).replace("\\", '')
        cursor.execute(sql)
        result = cursor.fetchall()
        if printResult:
            print(result)
        cursor.close()
        return result

    def updateData(self, sql):
        cursor = self.conn.cursor()
        print("执行sql---{}".format(sql))
        sql = str(sql).replace("\\", '')

        result = cursor.execute(sql)
        return result

    def saveFace128Vec(self, data):
        sql = "INSERT INTO `face_recogintion`.`known_face`(`face_name`, `face_encoding`) VALUES ('{}', '{}')".format(
            data["name"], data["face128vec"]
        )
        return self.updateData(sql)
        pass

    def getKnownFaceToDict(self):
        import numpy as np
        import json
        data = self.queryDataToDist("select * from known_face", printSql=False, printResult=False)
        known_face_encodings = []
        known_face_names = []

        for item in data:
            face_encoding = np.array(json.loads(str(item["face_encoding"])))
            known_face_encodings.append(face_encoding)
            known_face_names.append(item["face_name"])

        global known_face
        known_face["known_face_encodings"] = known_face_encodings
        known_face["known_face_names"] = known_face_names
        return known_face

    # 把数组/字典写进数据库
    def insertIntoTable(self, table, columns, data):
        """
        table: 要写入的数据表
        columns: 数据表列名
        data: 二维数组（列表）或者 字典
        """
        dataList = []
        column_str = ','.join(columns)

        # 传入参数是数组
        if isinstance(data, list):
            for d in data:
                # 处理数组中包含的空值
                arr = ['' if x is None else x for x in d]
                dataList.append(arr)

        # 传入参数是字典
        elif isinstance(data, dict):
            lis = list(data.values())
            for i in range(len(lis[0])):
                tempList = []
                for l in lis:
                    tempList.append(l[i])
                # 处理临时列表中的None值
                temp = ['' if x is None else x for x in tempList]
                dataList.append(temp)

        ret = 0
        try:
            cursor = self.conn.cursor()
            # 尝试写入sqlite数据库
            try:
                values = ('?,' * len(dataList[0]))[:-1]
                sqliteStr = 'insert into {}({}) values('.format(table, column_str) + values + ')'
                # 参数化执行，防止sql注入
                ret = cursor.executemany(sqliteStr, dataList)
            # 失败后写入mysql数据库
            except:
                values = ('%s,' * len(dataList[0]))[:-1]
                mysqlStr = 'insert into {}({}) values('.format(table, column_str) + values + ')'
                # 参数化执行，防止sql注入
                ret = cursor.executemany(mysqlStr, dataList)
            cursor.close()
        except Exception as e:
            print("写入数据库失败")
            print(e)
        return ret

    def close(self):
        self.conn.cursor.close()
        self.conn.close()
