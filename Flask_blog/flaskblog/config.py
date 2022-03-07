import os


class Config:
	SECRET_KEY='301653b4421209309537996ef97b19e5'
	SQLALCHEMY_DATABASE_URI='sqlite:///site.db' #Mysql로 바꾸기.
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

