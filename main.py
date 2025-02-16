import streamlit as st
import time

st.title('Streamlit 基本操作')

'スタート'

latest_iteration = st.empty()
bar = st.progress(30)

'スタート'

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Interation {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)

expander1 = st.expander('問い合わせ')
expander1.write('問い合わせ内容を書く')
expander2 = st.expander('問い合わせ')
expander2.write('問い合わせ内容を書く')
expander3 = st.expander('問い合わせ')
expander3.write('問い合わせ内容を書く')

