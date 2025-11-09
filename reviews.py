import streamlit as st
import oracledb

# db 연결 정보
DB_USER = "c##scottt"
DB_PASSWORD = "123456"
DB_HOST = "localhost"  
DB_PORT = 1521
DB_SERVICE_NAME = "xe"


# db 연결 
@st.cache_resource
def get_oracle_connection():
    """Oracle DB 연결 객체를 캐싱하여 재사용"""
    try:
        # oracledb.connect 호출을 캐싱 함수 내부에서 수행
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            service_name=DB_SERVICE_NAME
        )
        return connection
    except oracledb.Error as e:
        st.error(f"⚠️ Oracle DB 연결 실패: {e.args}")
        return None

# DB 연결 객체 생성 (get_oracle_connection 호출 시 연결이 이루어지거나 캐시된 객체가 반환됨)
conn = get_oracle_connection()



# 챗봇 대화를 db에 저장하는 함수
def save_chat_log(connection, user_q, classification_result):
    """챗봇 대화 로그를 Oracle DB에 저장"""
    if not connection:
        st.warning("데이터베이스 연결이 설정되지 않아 로그를 저장할 수 없습니다.")
        return

    try:
        cursor = connection.cursor()
        
        # CHATBOT_LOG_SEQ는 이전에 Oracle DB에 생성한 시퀀스 이름입니다.
        sql_query = """
        INSERT INTO CHATBOT_LOG (LOG_ID, USER_QUERY, CLASSIFICATION)
        VALUES (CHATBOT_LOG_SEQ.NEXTVAL, :user_q, :classification_result)
        """
        
        cursor.execute(sql_query, 
                       user_q=user_q, 
                       classification_result=classification_result)
        
        connection.commit()
        print("DB 로그 저장 완료") # VS Code 콘솔에 출력됨
        
    except oracledb.Error as e:
        error, = e.args
        st.error(f"데이터 저장 오류 발생: {error.message}")
        connection.rollback()
    finally:
        cursor.close()



## 모델 불러오기 
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
        st.error(f"모델 로드 실패! 폴더가 있는지 확인하세요: {e}")
        return None

classifier = load_model()

st.title('배달 앱 리뷰 감성 분류 봇 🤖')
st.write('파인튜닝된 KLUE/RoBERTa 모델로 리뷰를 긍정/부정 분류합니다.')

# 텍스트 영역 생성
user_query = st.text_area("리뷰를 여기에 입력하세요:", height=150)

with st.expander("예시 리뷰 보기"):
    st.write("👍 긍정 예시: 사장님이 너무 친절하시고 서비스도 좋아서 다음에도 꼭 주문하고 싶어요!")
    st.write("👎 부정 예시: 주문한 메뉴가 잘못 왔고, 포장이 엉망이라 다 식어서 왔네요.")


if st.button('분류하기'):
    if not classifier:
        st.warning("모델이 로드되지 않아 분류를 진행할 수 없습니다.")
    elif user_query.strip() == "":
        st.warning("분류할 리뷰 텍스트를 입력해주세요.")
    else:
        # 진행률 표시줄
        with st.spinner('리뷰 분석 중...'):

            # 예측 수행
            result = classifier(user_query)[0]

        label = result['label']
        score = result['score']

        # 결과 매핑
        sentiment = '긍정 👍' if label == 'LABEL_1' else '부정 👎'
        
        if label=='LABEL_1':
            classification_result= '긍정'
            if conn:
                save_chat_log(conn, user_query,  classification_result)
        elif label=='LABEL_0':
            classification_result= '부정'
            if conn:
                save_chat_log(conn, user_query, classification_result)
  

        st.success('✅ 분류 완료!')
        st.metric(label="분류 결과", value=sentiment)
        st.info(f"신뢰도: {score*100:.2f}%")

