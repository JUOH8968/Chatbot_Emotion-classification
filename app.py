import streamlit as st
from transformers import pipeline

# 저장된 모델 경로 (Colab 환경에서 my_emotional_classifier 폴더가 있어야 합니다)
MODEL_PATH = "ju03/Chatbot_Emotion-classification"   # "./my_emotional_classifier"

# 모델 로드 (앱이 로드될 때 한 번만 실행)
@st.cache_resource
def load_model():
    # 텍스트 분류 파이프라인 사용
    try:
        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH
        )
        return classifier
    except Exception as e:
        st.error(f"모델 로드 실패! 'my_emotional_classifier' 폴더가 있는지 확인하세요: {e}")
        return None

classifier = load_model()

st.title('배달 앱 리뷰 감성 분류 봇 🤖')
st.write('파인튜닝된 KLUE/RoBERTa 모델로 리뷰를 긍정/부정 분류합니다.')

# 텍스트 영역 생성
review_text = st.text_area("리뷰를 여기에 입력하세요:", height=150)

if st.button('분류하기'):
    if not classifier:
        st.warning("모델이 로드되지 않아 분류를 진행할 수 없습니다.")
    elif review_text.strip() == "":
        st.warning("분류할 리뷰 텍스트를 입력해주세요.")
    else:
        # 진행률 표시줄
        with st.spinner('리뷰 분석 중...'):

            # 예측 수행
            result = classifier(review_text)[0]

        label = result['label']
        score = result['score']

        # 결과 매핑
        sentiment = '긍정 👍' if label == 'LABEL_1' else '부정 👎'

        st.success('✅ 분류 완료!')
        st.metric(label="분류 결과", value=sentiment)
        st.info(f"신뢰도: {score*100:.2f}%")
