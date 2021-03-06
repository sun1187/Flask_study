from flask import Flask, request, jsonify
import json
import numpy as np
import pymysql
import torch
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from model import BERTDataset, BERTClassifier
#from recsys_step4 import all_rs
#from recsys_step3 import recsys_step3

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from config import Config

#db = SQLAlchemy()
#bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'users.login' #function name of route
login_manager.login_message_category = 'info'
#mail = Mail()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:my-secret-pw@127.0.0.1:3306/smtm"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = '301653b4421209309537996ef97b19e5'
db = SQLAlchemy(app)
#app.config.from_object(Config)
#db.init_app(app)
#bcrypt.init_app(app)
login_manager.init_app(app)
#mail.init_app(app)

#engine = db.create_engine("mysql+pymysql://root:my-secret-pw@127.0.0.1:3306/flask_test2")

# softmax 구하는 함수
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# device cpu로 불러오기 (local에서 돌리기 때문에)
device = torch.device("cpu")

# get_kobert
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# load model, map_location=device -> gpu로 모델을 저장하기 때문에 map_location을 통해 cpu로 다시 불러와야 오류가 안 남.
model = BERTClassifier(bertmodel)
model.load_state_dict(torch.load('/Users/eunsunkim/web_boaz/api_test/model_state_dict.pt', map_location=device))

label_softmax = []

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# model predict
def predict_result(predict_sentence):
  data = [predict_sentence, '0']
  dataset_another = [data]

  another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
  test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)
    
  model.eval()

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)

      valid_length= valid_length
      label = label.long().to(device)
      out = model(token_ids, valid_length, segment_ids)

      test_eval=[]
      for i in out:
          logits=i
          logits = logits.detach().cpu().numpy()
          logits = softmax(logits) # softmax

          label_softmax = logits
          if np.argmax(logits) == 0:
                test_eval.append("기쁨")
          elif np.argmax(logits) == 1:
                test_eval.append("슬픔")
          elif np.argmax(logits) == 2:
                test_eval.append("분노")
          elif np.argmax(logits) == 3:
                test_eval.append("중립")
                pass

      return label_softmax, np.argmax(logits)


#{"values":"가나다라"}
@app.route('/predict', methods=['POST'])
def predict():
    event = json.loads(request.data)#, encoding="UTF-8")
    values = event['values']
    print(values)
    nlp_out = predict_result(values)
    print(nlp_out)
    print('nlp 결과 출력', nlp_out[0][:3])
    return str(nlp_out[0][:3])

#{"values":"가나다라",
#"emotion":0.3,
#"lyric":0.7}
@app.route('/predict_with_music_step3', methods=['POST'])
def predict_with_music_step3():
    event = json.loads(request.datas)#, encoding="UTF-8")
    values = event['values']
    emo_weight = event['emotion']
    lyric_weight = event['lyric']
    print(values)
    print(emo_weight)
    print(lyric_weight)
    nlp_out = predict_result(values)
    print(nlp_out)
    weights = [emo_weight, lyric_weight]
    result = recsys_step3(nlp_out[0][:3], values, weights)
    #print(result)
    result.to_csv('fin_result.csv', encoding='utf-8-sig')

    #추천 노래
    first = [result['곡 제목'][0], result['가수'][0], result['원가사'][0], result['tag max'][0], result['대분류str'][0]]
    second= [result['곡 제목'][1], result['가수'][1], result['원가사'][1], result['tag max'][1], result['대분류str'][1]]
    third = [result['곡 제목'][2], result['가수'][2], result['원가사'][2], result['tag max'][2], result['대분류str'][2]]
    fourth = [result['곡 제목'][3], result['가수'][3], result['원가사'][3], result['tag max'][3], result['대분류str'][3]]
    fifth = [result['곡 제목'][4], result['가수'][4], result['원가사'][4], result['tag max'][4], result['대분류str'][4]]
    df = [first, second, third, fourth, fifth]

    return jsonify({
        "확률": str(nlp_out[0][:3]),
        "top1_name": df[0][0],
        "top1_singer": df[0][1],
        "top2_name": df[1][0],
        "top2_singer": df[1][1],
        "top3_name": df[2][0],
        "top3_singer": df[2][1],
        "top4_name": df[3][0],
        "top4_singer": df[3][1],
        "top5_name": df[4][0],
        "top5_singer": df[4][1]
    })


#{"emotion_list": [1.6410377e-04, 9.9766809e-01, 2.1216171e-03],
#"emt_ch": 1,
#"scores": [4, 2, 1, 3, 5]
#}
# 이름 잘 짓기. swagger api 문서 작성
@app.route('/re_recys', methods=['POST'])
def re_recys():
    event = json.loads(request.data)#, encoding="UTF-8")
    emotion_list = event['emotion_list']
    emt_ch = event['emt_ch']
    scores = event['scores']
    result = all_rs(emotion_list, emt_ch, scores)

    # 추천 노래
    first = [result['곡 제목'][0], result['가수'][0], result['원가사'][0], result['tag max'][0], result['대분류str'][0]]
    second = [result['곡 제목'][1], result['가수'][1], result['원가사'][1], result['tag max'][1], result['대분류str'][1]]
    third = [result['곡 제목'][2], result['가수'][2], result['원가사'][2], result['tag max'][2], result['대분류str'][2]]
    fourth = [result['곡 제목'][3], result['가수'][3], result['원가사'][3], result['tag max'][3], result['대분류str'][3]]
    fifth = [result['곡 제목'][4], result['가수'][4], result['원가사'][4], result['tag max'][4], result['대분류str'][4]]
    df = [first, second, third, fourth, fifth]

    return jsonify({
        "top1_name": df[0][0],
        "top1_singer": df[0][1],
        "top2_name": df[1][0],
        "top2_singer": df[1][1],
        "top3_name": df[2][0],
        "top3_singer": df[2][1],
        "top4_name": df[3][0],
        "top4_singer": df[3][1],
        "top5_name": df[4][0],
        "top5_singer": df[4][1]
    })

#회원 가입(post), 로그인(post), 작성한 글 목록(get), 글(세부내용)(get)
#조회: get, 내보내기: post
from flask import session
from forms import UserLoginForm
from flask_model import user_table, diary_table, recommend_table
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()
bcrypt.init_app(app)

@app.route('/login', methods=['POST'])
def login():
    user_name = request.form.get("user_name")
    password = request.form.get("password")
    #form = UserLoginForm()
    user = user_table.query.filter_by(user_name = user_name).first()
    if not user:
        return "존재하지 않는 사용자이다."
    if not bcrypt.check_password_hash(user.password, password):
        return "비밀번호 틀렸다."
    return jsonify({
        "user_id": user.user_id,
        "user_name": user.user_name,
        "user_nickname": user.user_nickname,
        "password": user.password,
    })

#table에 modify update 수정하기
#diary_id -> long or bigint
#

@app.route('/register', methods=['POST'])
def register():
    #form = RegistrationForm()
    user_name = request.form.get("user_name")
    user_nickname = request.form.get("user_nickname")
    password = request.form.get("password")
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    user = user_table(user_name=user_name, user_nickname=user_nickname, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    return "회원가입 성공"

@app.route("/post/new", methods=['POST'])
def new_post():
    diary_name = request.form.get("diary_name")
    diary_date = request.form.get("diary_date")
    content = request.form.get("content")
    happy_score = request.form.get("happy_score")
    sad_score = request.form.get("sad_score")
    angry_score = request.form.get("angry_score")
    mid_score = request.form.get("mid_score")
    user_id = request.form.get("user_id") ####나중에 고칠필요. 로그인 시 기능으로.
    post = diary_table(diary_name=diary_name, diary_date=diary_date,
                       content=content, happy_score=happy_score,
                       sad_score=sad_score, angry_score=angry_score, mid_score=mid_score,
                       user_id=user_id)
    db.session.add(post)
    db.session.commit()
    return '글 쓰기 성공'

@app.route("/post/<int:post_id>")
def post(post_id):
    post = diary_table.query.get_or_404(post_id)
    return jsonify({
        "diary_id": post.diary_id,
        "diary_name": post.diary_name,
        "diary_date": post.diary_date,
        "content": post.content,
        "user_id": post.user_id,
    })


@app.route("/user/<string:user_name>")
def user_posts(user_name):
    page = request.args.get('page', 1, type=int)
    user = user_table.query.filter_by(user_name=user_name).first_or_404()

    posts = diary_table.query.filter_by(user_id=user.user_id)\
        .order_by(diary_table.create_date.desc())\
        .paginate(page=page, per_page=5, error_out=False)

    post_df = []
    for ele in posts.items:
        Data = {'id': ele.diary_id, 'diary_name': ele.diary_name, 'content': ele.content}
        post_df.append(Data)

    return jsonify({'key': post_df})


@app.route("/post/save_rec_result", methods=['POST'])
def save_rec_result():
    diary_id = request.form.get("diary_id")
    song_sequence = request.form.get("song_sequence")
    rec_song_id = request.form.get("rec_song_id")
    rec_score = request.form.get("rec_score")
    rec_result = recommend_table(diary_id=diary_id, song_sequence=song_sequence,
                       rec_song_id=rec_song_id, rec_score=rec_score)
    db.session.add(rec_result)
    db.session.commit()
    return '노래 저장 성공'
# 글쓰면 db insert
# 노래 정보 저장
#

#    if form.validate_on_submit():
#        error = None
#        user = User.query.filter_by(username=form.username.data).first()
#        print(user)
#        print(user.passowrd)
#        if not user:
#            return "존재하지 않는 사용자입니다."
#        elif not check_password_hash(user.password, form.password.data):
#            return "비밀번호가 올바르지 않습니다."
        #if error is None:
        #    session.clear()
        #    session['user_id'] = user.id
        #    return redirect(url_for('main.index'))
        #flash(error)
#        return "로그인 성공"
#    return "아무것도 없음"


if __name__ == '__main__':
    app.run(debug=True)

