import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
import joblib

fires = pd.read_csv("./sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

# 1. 데이터 상위 5개 행 출력
print("fires.head() 결과:")
print(fires.head())

# 2. 데이터프레임 전체 정보 확인 (컬럼, 결측치, 데이터 타입 등)
print("\n fires.info() 결과:")
fires.info()

# 3. 수치형 데이터 통계 요약 (평균, 표준편차, 사분위 등)
print("\n fires.describe() 결과:")
print(fires.describe())

# 4. month 카테고리 분포
print("\n month value_counts() 결과:")
print(fires["month"].value_counts())

# 5. day 카테고리 분포
print("\n day value_counts() 결과:")
print(fires["day"].value_counts())

fires.hist(bins=50, figsize=(12, 8))
plt.show()

plt.scatter(fires["temp"], fires["burned_area"], alpha=0.5)
plt.xlabel("temp")
plt.ylabel("burned_area (log)")
plt.show()

1-5
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
print("test_set 비율 확인:", len(test_set) / len(fires))
print(test_set.head())
fires["month"].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nMonth category proportion: \n", strat_test_set["month"].value_counts()/len(strat_test_set))
print("\nOverall month category proportion: \n", fires["month"].value_counts()/len(fires))

# 1-6
# 사용할 주요 수치형 특성들 선택
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]

# scatter matrix 출력
scatter_matrix(fires[attributes], figsize=(12, 8), alpha=0.5)
plt.suptitle(" Scatter Matrix", fontsize=16)
plt.show()


# 1-7 산불 발생 지역 시각화 (위도/경도 기준)
fires.plot(kind="scatter",
           x="longitude", y="latitude",
           alpha=0.4,
           s=fires["max_temp"],         # 원 크기: 최대 온도
           label="max_temp",
           c="burned_area",             # 색상: 피해 면적
           cmap=plt.get_cmap("jet"),    # 색상 맵: jet
           colorbar=True,
           figsize=(10, 7))

plt.title("Number of forest fires by region (color: area affected, size: temperature)")
plt.legend()
plt.show()

# 1-8 
# month, day만 추출
cat_day = fires[["day"]]
cat_month = fires[["month"]]

# 인코더 생성 및 학습
cat_day_encoder = OneHotEncoder()
cat_month_encoder = OneHotEncoder()

cat_day_1hot = cat_day_encoder.fit_transform(cat_day)
cat_month_1hot = cat_month_encoder.fit_transform(cat_month)

# 희소 행렬 그대로 출력 (마지막 10개만 보기 좋게)
print(cat_day_1hot[-10:])     # 
print(cat_month_1hot[-10:])

# 카테고리 확인
print("\ncat_day_encoder.categories_ :")
print(cat_day_encoder.categories_)

print("\ncat_month_encoder.categories_ :")
print(cat_month_encoder.categories_)

# 레이블 분리
fires = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()

# 수치형 / 범주형 특성 나누기
num_attribs = list(fires.drop(["month", "day"], axis=1))
cat_attribs = ["month", "day"]

# 수치형 파이프라인 (표준화)
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# 전체 파이프라인 구성
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

# 전처리 수행
fires_prepared = full_pipeline.fit_transform(fires)
joblib.dump(full_pipeline, "pipeline.pkl")

# 결과 확인
print("\n\n########################################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")
print(fires_prepared.shape)

# Keras model 개발
fires_test = strat_test_set.drop("burned_area", axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()

fires_test_prepared = full_pipeline.transform(fires_test)

X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

# Keras 모델 저장
model.save('fires_model.keras')

# evaluate model
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))
