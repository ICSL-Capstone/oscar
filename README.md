
# 레포지토리 설치
레포지토리 설치한 위치를 잘 확인할 것.
```
pwd
```
를 입력하여 설치하는 위치를 잘 기억해둔 다음 아래 명령어 실행.
```
git clone https://github.com/ICSL-Capstone/oscar --recursive
```
<br>

# 모델 학습 방법

```sh
python neural_net/train.py 데이터셋_경로
```
<br>


## example 
---
데이터셋의 폴더이름이 2021-05-18-15-25-54 이고  
본인이 받은 데이터셋의 경로가 /home/icsl/dataset/ 에 있을때,  
csv 파일의 경로는 
```sh
/home/icsl/dataset/2021-05-18-15-25-54/2021-05-18-15-25-54.csv
```
처럼 될것임.  <br>

이때 아래와 같이 csv가 들어있는 폴더 경로를 입력해주면 됨.  
```sh
python neural_net/train.py /home/icsl/dataset/2021-05-18-15-25-54
```

### #주의#
위 명령어를 실행하는 위치는 oscar 폴더(설치한 레포지토리) 내에서 실행해야함.  
즉, 터미널창에
```sh
pwd
```
입력시 나오는 경로가 레포지토리가 설치된 위치가 맞는지 확인할 것.

만약 현재 위치가 레포지토리가 설치된 위치가 아니라면, cd 명령어로 설치한 폴더로 이동
```sh
cd oscar_레포지토리가_설치된_경로
```
ex)
```sh
cd /home/kdh/oscar
```


---