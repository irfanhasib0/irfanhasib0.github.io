#jupyter nbconvert --to markdown html README.ipynb 
#cp README.md README.html
jupyter nbconvert --to html README.ipynb 
#cp README.html index.html
cp README.html app/public/readme.html
cp *css app/public/
#cd app && npm run start
