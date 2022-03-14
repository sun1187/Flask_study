import os
import pymysql
import mysql.connector
pymysql.install_as_MySQLdb()

db = {
    'user'     : 'user2', #'root',
    'password' : '+"test"+', #'+"my-secret-pw"+',
    'host'     : '127.17.0.1',
    'port'     : '3306',
    'database' : 'flask_test'
}

class Config:
	SECRET_KEY='301653b4421209309537996ef97b19e5'
	#SQLALCHEMY_DATABASE_URI='sqlite:///site.db' #Mysql로 바꾸기.
	#SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"
	SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://root:my-secret-pw@127.0.0.1:3306/flask_test2"
	MAIL_SERVER='smtp.googlemail.com'
	MAIL_PORT=587
	MAIL_USE_TLS=True
	MAIL_USERNAME=os.environ.get('EMAIL_USER')
	MAIL_PASSWORD=os.environ.get('EMAIL_PASS')

#api 통신만한다.
#파이썬에서 request나 post 요청. 프론트도 가능.

# 모델 연동
#필요한 파라미터랑 필요한 명령어. 
#flask로 만들고 postman으로 테스트 

