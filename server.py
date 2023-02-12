from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from Data import Data

from search import Search

app = Flask(__name__)
data = Data()
Search = Search()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        if not file :
            return

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = data.uploaded_path+'/' +datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search

        ids, dists = Search.search(img)
        paths = [data.imgs_path+'/'+str(id)+'.jpg' for id in ids]
        scores = [x for x in zip(dists,paths)]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
