cd app && npm run build
cp -r ./app/build/* ../docs/
git add ../docs
git commit -m "deploying new version"
git push origin master
