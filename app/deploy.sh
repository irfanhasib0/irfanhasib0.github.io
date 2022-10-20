npm run build
cp -r ./build/* ../docs/
git add ../docs
git commit -m "deploying new version"
git push origin master
