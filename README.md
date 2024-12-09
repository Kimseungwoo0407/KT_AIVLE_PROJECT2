---
## 프로젝트 목적🎯
- 이 프로젝트는 타깃 변수를 포함한 데이터셋을 생성하여 MT5 모델의 파인튜닝을 하는 것을 목표로 합니다.
- 텍스트 요약 및 평가 시스템을 구축하고, 생성된 고품질 요약을 MT5 모델의 파인튜닝 데이터셋으로 활용할 수 있도록 합니다.
- 구체적으로, 다음과 같은 목적을 가지고 진행됩니다
---
### 1. MT5 모델 파인튜닝을 위한 고품질 요약 데이터셋 생성
- TextRank 알고리즘을 사용하여 텍스트에서 중요한 문장을 추출하고, 이를 요약합니다.
- 요약된 텍스트는 MT5 모델의 파인튜닝 데이터셋으로 사용될 수 있습니다.
---
### 2. 요약 품질 평가 및 선택
- 각 요약은 ROUGE-1 F1 점수를 사용하여 품질을 평가합니다.
- ROUGE 점수가 일정 수준(예: 0.7) 이상인 요약만을 고품질 데이터셋으로 필터링하여 확보합니다.
- 이 고품질 요약 데이터는 MT5 모델이 실제 사용되는 환경에서 높은 정확도로 작동할 수 있도록 학습에 사용됩니다.
---
### 3.MT5 파인튜닝을 위한 CSV 형식의 데이터셋 저장
- 요약된 데이터와 그에 대한 평가 정보를 CSV 파일로 저장하여, MT5 모델 학습에 활용할 수 있는 형식으로 제공합니다.
- 이 데이터셋은 추후 MT5 모델 파인튜닝에 활용되며, 모델의 성능 향상에 기여할 것입니다.
---
## 프로젝트 실행 방법🚀
---
### 1.필수 라이브러리 설치
필요한 라이브러리를 설치하려면 아래 명령어를 실행하세요.
```bash
pip install transformers datasets evaluate rouge_score
pip install transformers[torch]
pip install accelerate -U
!pip install transformers==4.30
```
---
### 2.데이터셋 준비 및 전처리
- filtered_data.csv 파일을 Google Drive에서 불러와 데이터셋을 생성하고, 텍스트 전처리를 통해 요약에 필요한 형식으로 데이터를 준비합니다.
- Dataset.from_pandas를 사용하여 pandas 데이터프레임을 Hugging Face Dataset 객체로 변환합니다.
- 각 데이터는 Answer 열을 입력 텍스트로, summary 열을 요약 텍스트로 사용합니다.
---
### 3.데이터 전처리
- preprocess_function을 통해 각 텍스트에 summarize: 접두어를 붙여 모델 입력 형식에 맞게 전처리합니다.
- max_length를 설정하여 토큰화된 입력 텍스트의 길이를 제한하고, labels는 요약 텍스트로 설정하여 모델 훈련에 필요한 데이터를 준비합니다.
---
### 4.모델 및 토크나이저 준비
- Google MT5 모델을 사용하여 요약을 생성합니다. "google/mt5-small" 모델을 로드하고, AutoTokenizer로 텍스트를 토크나이즈합니다.
```python
checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
---
### 5.데이터 준비 및 학습
- tokenized_dataset을 통해 학습 데이터셋과 테스트 데이터셋을 나누고, DataCollatorForSeq2Seq로 데이터를 배치 처리합니다.
- Seq2SeqTrainer를 사용하여 모델 훈련을 시작합니다.
- train_dataset과 eval_dataset을 각각 훈련 및 평가용 데이터로 지정하고, compute_metrics 함수로 ROUGE 점수를 계산하여 성능을 평가합니다.
---
### 6.모델 훈련
- 모델은 Seq2SeqTrainingArguments를 통해 훈련 파라미터를 설정하고, trainer.train()을 호출하여 훈련을 진행합니다.
```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
```
---
### 7.훈련된 모델 푸쉬 및 요약 생성
- 훈련이 완료된 모델은 Hugging Face Hub에 업로드하여 모델을 공유할 수 있습니다.
```python
trainer.push_to_hub()
```
- 훈련된 모델을 사용하여 텍스트를 요약할 수 있습니다.
- 예를 들어, 아래와 같이 pipeline을 사용하여 새로운 텍스트의 요약을 생성할 수 있습니다

```python
summarizer = pipeline(
    "summarization",
    model="aoome123/my_model",
    max_length=600,
    no_repeat_ngram_size=3,
    min_length=150,
    length_penalty=2.0,
    num_beams=8
)

text = """학부 3학년부터 건설/건축 엔지니어가 되기 위한 목표를 세웠고..."""  # 요약할 텍스트

summarizer(text)
```
---
### 8.결과 및 평가
- ROUGE 지표를 사용하여 모델의 요약 품질을 평가하며, compute_metrics 함수는 ROUGE 점수와 요약 길이를 출력합니다.
- evaluate.load("rouge")를 통해 ROUGE 지표를 계산하고, predictions와 labels를 비교하여 성능을 측정합니다.
---
