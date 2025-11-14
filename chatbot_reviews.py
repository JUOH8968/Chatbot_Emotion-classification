import streamlit as st
from transformers import pipeline

# --- 모델 로드 함수 (기존 코드와 동일) ---
MODEL_PATH = "ju03/Chatbot_Emotion-classification" 

@st.cache_resource
def load_model():
    # 텍스트 분류 파이프라인 사용
    try:
        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH
        )
        st.success("감정 분류 모델 로드 완료!")
        return classifier
    except Exception as e:
        st.error(f"❌ **모델 로드 실패!** Hugging Face 모델을 다운로드하거나 초기화하는 데 문제가 발생했습니다. ({e})")
        return None

classifier = load_model()

# --- Streamlit UI 시작 ---
st.title('배달 어플 리뷰 감정 분류 봇 🤖')
st.write('파인튜닝된 KLUE/RoBERTa 모델로 리뷰를 긍정/부정 분류합니다.')

with st.expander("예시 리뷰 보기"):
    st.write("👍 긍정 예시: 사장님이 너무 친절하시고 서비스도 좋아서 다음에도 꼭 주문하고 싶어요!")
    st.write("👎 부정 예시: 주문한 메뉴가 잘못 왔고, 포장이 엉망이라 다 식어서 왔네요.")



# 1. 채팅 기록 저장을 위한 session_state 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    # 초기 봇 메시지 추가 (선택 사항)
    st.session_state["messages"].append(
        {"role": "assistant", "content": "안녕하세요! 배달 어플 리뷰를 입력하시면 긍정인지 부정인지 분류해 드립니다."}
    )

# 2. 기존 채팅 기록 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력 처리 (st.chat_input 사용)
# 사용자가 새로운 리뷰를 입력했을 때만 실행됩니다.
if prompt := st.chat_input("리뷰를 문장으로 여기에 입력하세요."):
    if not classifier:
        st.warning("모델이 로드되지 않아 분류를 진행할 수 없습니다.")
    else:
        # 3-1. 사용자 메시지(리뷰)를 세션 상태에 저장하고 화면에 표시
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3-2. 모델을 사용하여 감성 분류 진행
        with st.spinner('리뷰 분석 중...'):
            try:
                result = classifier(prompt)[0]
                label = result['label']
                score = result['score']

                # 결과 매핑
                sentiment_emoji = '긍정 👍' if label == 'LABEL_1' else '부정 👎'
                
                # 3-3. 봇의 답변 생성
                bot_response = f"""
                **[분석 결과]**
                - **감정:** {sentiment_emoji}
                - **신뢰도:** {score*100:.2f}%
                """

                # 3-4. 봇의 답변을 세션 상태에 저장하고 화면에 표시
                st.session_state["messages"].append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
                
            except Exception as e:
                error_message = f"❌ **리뷰 분류 중 오류 발생!** 오류 상세: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):

                    st.error(error_message)


