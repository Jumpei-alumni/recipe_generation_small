import streamlit as st
from model import RecipeBART
from transformers import BartTokenizer
import torch


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

st.title('Automatic Recipe Generation')
st.write('自動料理手順生成')
data_load_state = st.text('loading....（準備中）')
model = RecipeBART()
model.load_state_dict(torch.load('bart_recipe_small_cpu.pth'))
data_load_state = st.text('ready to start!!（準備完了！！）')

st.sidebar.write('sampling parameter(サンプリング手法)')
tmp = st.sidebar.slider("tmperature", min_value=0.5, max_value=1.0, step=0.01, value=1.0)
top_k = st.sidebar.slider("top-k", min_value=2, max_value=100, step=1, value=40)

left_column, right_column = st.columns(2)
name = left_column.text_input('cooking name(料理名)', 'vegetable soup meatloaf')
ingredients = right_column.text_input('ingredients you want to use(利用したい材料)', 'onion,salt')
right_column.caption('Enter the ingredients separated by ,(,で区切って入力してください)')

make_recipe_button = st.button('make recipe(料理手順生成)')


if make_recipe_button:
    if (ingredients.replace(' ','') != '') and (name.replace(' ','') != ''):
        data_load_state = st.text('Loading data...(頑張って作っています)')
        ingredients = ingredients.replace(', ', ',').replace(' ,', ',')
        recipe = 'aa'
        n = 0
        while len(recipe.replace('<s>', '').replace('</s>', '').split('>')) != 2:
            recipe = tokenizer.decode(model.generate(name, ingredients, False, top_k=top_k, tmp=tmp))
            n += 1
            if n > 4:
                st.write('please try again(もう一度お試しください)')
                break
        recipe = recipe.replace('<s>', '').replace('</s>', '').split('>')
        if len(recipe) == 2:
            data_load_state = st.text('Done!!(できた！！)')
            ingr = ', '.join(list(set(recipe[0].split(','))))
            step = recipe[1]
            st.subheader('required ingredients（必要材料）')
            st.write(ingr)
            st.subheader('cooking steps（料理手順）')
            st.write(step)
            
            #r = 'required ingredients\n'+ingr+'\ncooking steps\n'+step
            #url = 'https://www.deepl.com/ja/translator#en/ja/'+r
            #st.components.v1.html('<a href="'+url+'" class="btn">翻訳</a>')
            
            
    else:
        st.write('Please fill in the blanks.(空欄を埋めてください)')