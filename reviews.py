import streamlit as st
import oracledb
from transformers import pipeline

# 🚨 중요: 여기에 실제 테이블/시퀀스를 소유한 스키마 이름을 넣어주세요.
TABLE_NAME = "BOT_REVIEW_LOG"

# db 연결
# DB 연결 객체를 캐싱하여 재사용
def get_oracle_connection():
    """Streamlit Secrets에서 DB 정보를 가져와 Oracle DB 연결 객체를 캐싱하여 재사용"""
    
    # st.secrets에서 [db_credentials] 섹션의 값을 읽어옵니다.
    try:
        connection = oracledb.connect(
            user=st.secrets["db_credentials"]["user"],
            password=st.secrets["db_credentials"]["password"],
            host=st.secrets["db_credentials"]["host"],
            port=int(st.secrets["db_credentials"]["port"]), 
            sid=st.secrets["db_credentials"]["sid"]
        )
        st.success("✅ Oracle DB 연결 성공! (Secrets 사용)")
        return connection
    
    except KeyError:
        st.error("⚠️ **Secrets 파일에 DB 연결 정보가 누락되었습니다.** `secrets.toml`의 `[db_credentials]` 섹션을 확인하세요.")
        return None
        
    except ValueError as e: 
        st.error(f"⚠️ **포트 값 변환 오류:** 포트 값은 정수여야 합니다. Secrets 파일의 포트 값을 확인하세요. 상세 오류: {e}")
        return None
        
    except oracledb.Error as e:
        error_obj = e.args[0]
        st.error(f"❌ **Oracle DB 연결 실패**")
        st.error(f"오류 코드: **{error_obj.code}**")
        st.error(f"오류 메시지: **{error_obj.message}**")
        return None

# DB 연결 객체 생성
conn = get_oracle_connection()


# 챗봇 대화를 db에 저장하는 함수
def save_chat_log(connection, user_q, classification_result):
    """챗봇 대화 로그를 Oracle DB에 저장"""
    cursor = None
    status_message = ""
    if not connection:
        status_message = "⚠️ 데이터베이스 연결이 설정되지 않아 로그를 저장할 수 없습니다."
        return status_message

    try:
        cursor = connection.cursor()

        # 스키마, 테이블, 시퀀스 이름을 모두 큰따옴표로 명시하여 정확한 참조를 보장합니다.
        sql_query = f"""
        INSERT INTO {TABLE_NAME} (LOG_ID, USER_QUERY, CLASSIFICATION)
        VALUES ({TABLE_NAME}_SEQ.NEXTVAL, :user_q, :classification_result)
        """
        
        # 바인딩 변수 사용: SQL Injection 위험을 줄이고 데이터 타입 안정성을 높입니다.
        cursor.execute(sql_query, 
                        user_q=user_q, 
                        classification_result=classification_result)
        
        connection.commit()
        status_message = f"💾 DB 테이블 '{TABLE_NAME}'에 성공적으로 저장되었습니다."
        
    except oracledb.Error as e:
        error, = e.args
        if error.code == 942:
            status_message = f"⚠️ DB 로그 저장 실패! 테이블/시퀀스 '{TABLE_NAME}'이 존재하지 않거나 권한이 없습니다. (Code: {error.code})"
        elif error.code == 2289:
            status_message = f"⚠️ DB 로그 저장 실패! 시퀀스 '{TABLE_NAME}_SEQ'가 존재하지 않습니다. 시퀀스를 생성해야 합니다. (Code: {error.code})"
        else:
            status_message = f"⚠️ 데이터 저장 오류 발생: [Code: {error.code}] {error.message}"
        connection.rollback()
    
    finally:
        if cursor:
            cursor.close()
            
    # st.info(status_message) # 로그 저장 상태를 확인하고 싶을 때 주석 해제

    return status_message


## 모델 불러오기 
# 저장된 모델 경로 
MODEL_PATH = "ju03/Chatbot_Emotion-classification" 

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
        st.success("🧠 감성 분류 모델 로드 완료!")
        return classifier
    except Exception as e:
        st.error(f"❌ **모델 로드 실패!** Hugging Face 모델을 다운로드하거나 초기화하는 데 문제가 발생했습니다. 상세 오류를 콘솔에서 확인하세요. ({e})")
        return None

classifier = load_model()

# --- Streamlit UI 시작 ---
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
            try:
                # ----------------------------------------------------
                # ✅ 주요 수정 사항: 모델 추론을 try-except로 감싸 오류 포착
                # ----------------------------------------------------
                result = classifier(user_query)[0]
                
                label = result['label']
                score = result['score']

                # 결과 매핑
                sentiment = '긍정 👍' if label == 'LABEL_1' else '부정 👎'
                classification_result = '긍정' if label == 'LABEL_1' else '부정'
                
                # DB 로깅
                if conn:
                    # conn이 연결되어 있으면, 분류 결과(긍정/부정)를 담아 로그 저장
                    log_status = save_chat_log(conn, user_query, classification_result)
                    st.info(log_status) # 선택 사항: 로그 저장 상태 표시

                # 결과 출력
                st.success('✅ 분류 완료!')
                st.metric(label="분류 결과", value=sentiment)
                st.info(f"신뢰도: {score*100:.2f}%")
                
            except Exception as e:
                # 오류 발생 시 사용자에게 명확하게 알림
                st.error(f"❌ **리뷰 분류 중 심각한 오류가 발생했습니다.**")
                st.error(f"오류 상세: {e}")
                st.warning("Streamlit을 재시작하거나, 콘솔 창에서 상세 오류 로그를 확인해 주세요.")
