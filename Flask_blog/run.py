from flaskblog import create_app #__init__.py 실행

app = create_app()

if __name__ == '__main__':
	app.run(debug=True)

