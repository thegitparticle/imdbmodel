import streamlit as st
import PIL
from fastai import *
from fastai.text import *
from pathlib import Path

path = Path(__file__).parent

@st.cache(allow_output_mutation = True)
def learner(path):						#load the learner from 'export.pkl'
	learn = Learner.load(path)	
	return learn


def main():
	st.set_option('deprecation.showfileUploaderEncoding', False)
	html_title = """  
	<div style="text-align:center;"> 
		<h1>Praise or Criticism</h1>
	</div>
	"""
	st.markdown(html_title, unsafe_allow_html = True)
	st.subheader("type something below and see if a bunch of matrices can understand you!")
	learn = learner(path)
	written_text = st.file_uploader("type something...")
	if written_text is not None:
		st.text(written_text)
		pred = learn.predict(written_text)	
		st.success(pred)
	footer = """
	<div style="position:fixed; text-align:center; bottom:0px; right:0px; left:0px; background-color:black" markdown="1">
		<h4>Built with Streamlit using <a href= "https://www.fast.ai">fast.ai</a></h4>
		<p><a href="https://github.com/apzl/mask-or-not">Github</a> | &copyapsal</p>
	</div>
	"""
	st.markdown(footer, unsafe_allow_html=True)

if __name__=='__main__':
    main()