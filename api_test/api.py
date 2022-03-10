from flask import Flask, request
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from model import BERTDataset, BERTClassifier
from recsys_step4 import all_rs
from recsys_step3 import recsys_step3

app = Flask(__name__)

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
    event = json.loads(request.data)#, encoding="UTF-8")
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

    print()
    print('**Top1')
    print('곡 제목:', df[0][0])
    print('가수:', df[0][1])
    #print('원가사:', df[0][2])
    #print('tag max:', df[0][3])
    #print('대분류str:', df[0][4])
    print()
    print('**Top2')
    print('곡 제목:', df[1][0])
    print('가수:', df[1][1])
    #print('원가사:', df[1][2])
    #print('tag max:', df[1][3])
    #print('대분류str:', df[1][4])
    print()
    print('**Top3')
    print('곡 제목:', df[2][0])
    print('가수:', df[2][1])
    #print('원가사:', df[2][2])
    #print('tag max:', df[2][3])
    #print('대분류str:', df[2][4])
    print()
    print('**Top4')
    print('곡 제목:', df[3][0])
    print('가수:', df[3][1])
    #print('원가사:', df[3][2])
    #print('tag max:', df[3][3])
    #print('대분류str:', df[3][4])
    print()
    print('**Top5')
    print('곡 제목:', df[4][0])
    print('가수:', df[4][1])
    #print('원가사:', df[4][2])
    #print('tag max:', df[4][3])
    #print('대분류str:', df[4][4])

    return str(nlp_out[0][:3])


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

    print()
    print('**Top1')
    print('곡 제목:', df[0][0])
    print('가수:', df[0][1])
    # print('원가사:', df[0][2])
    # print('tag max:', df[0][3])
    # print('대분류str:', df[0][4])
    print()
    print('**Top2')
    print('곡 제목:', df[1][0])
    print('가수:', df[1][1])
    # print('원가사:', df[1][2])
    # print('tag max:', df[1][3])
    # print('대분류str:', df[1][4])
    print()
    print('**Top3')
    print('곡 제목:', df[2][0])
    print('가수:', df[2][1])
    # print('원가사:', df[2][2])
    # print('tag max:', df[2][3])
    # print('대분류str:', df[2][4])
    print()
    print('**Top4')
    print('곡 제목:', df[3][0])
    print('가수:', df[3][1])
    # print('원가사:', df[3][2])
    # print('tag max:', df[3][3])
    # print('대분류str:', df[3][4])
    print()
    print('**Top5')
    print('곡 제목:', df[4][0])
    print('가수:', df[4][1])
    # print('원가사:', df[4][2])
    # print('tag max:', df[4][3])
    # print('대분류str:', df[4][4])

    return "-1"

#회원 가입(post), 로그인(post), 작성한 글 목록(get), 글(세부내용)(get)
#조회: get, 내보내기: post


if __name__ == '__main__':
    app.run(debug=True)

