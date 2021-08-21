import pandas as pd
import streamlit as st

from MLUtils import *

st.title("Dr. Z.C.M.")
st.title('AI and PONV')  # 算法名称 and XXX

vars = []

uploader = None
btn_predict = None

lsvc = None
adab = None


# 配置选择变量（添加生成新数据并预测的功能）
def setup_selectors():
    global vars, uploader, btn_predict

    if uploader is not None and COL_INPUT is not None and len(COL_INPUT) > 0:
        col_num = 3
        cols = st.columns(col_num)

        for i, c in enumerate(COL_INPUT):
            with cols[i % col_num]:
                num_input = st.number_input(f"Please input {c}", value=0, format="%d", key=c)
                vars.append(num_input)

        with cols[0]:
            btn_predict = st.button("Do Predict")

    if btn_predict:
        do_predict()


# 配置上传条
def setup_uploader():
    global uploader
    uploader = st.file_uploader('Upload data file here', type=['csv', 'txt'], help='Upload dataset for training',
                                key="uploader")

    if uploader is not None:
        if st.checkbox('Show uploaded file information', value=True, key="cb_file_info"):
            # 显示文件详细信息
            file_details = {
                "FileName": uploader.name, "FileType": uploader.type, "FileSize": uploader.size
            }
            st.write(file_details)

        do_processing()
        setup_selectors()


# 对上传的文件进行处理和展示
def do_processing():
    global uploader
    global lsvc, adab
    pocd = read_csv(uploader)

    st.text("Dataset Description")
    st.write(pocd.describe())
    if st.checkbox('Show detail of this dataset'):
        st.write(pocd)

    # 分割数据
    X_train, X_test, y_train, y_test = do_split_data(pocd)
    X_train, X_test, y_train, y_test = do_xy_preprocessing(X_train, X_test, y_train, y_test)

    col1, col2 = st.columns(2)

    # 准备模型
    lsvc = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                     fit_intercept=True,
                     intercept_scaling=1, class_weight=None, verbose=0, random_state=1, max_iter=1000)
    lsvc = CalibratedClassifierCV(lsvc)
    adab = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=1)

    # 模型训练、显示结果
    with st.spinner("Training, please wait..."):
        lsvc_result = model_fit_score(lsvc, X_train, y_train)
        adab_result = model_fit_score(adab, X_train, y_train)
    with col1:
        st.text("Training Result")
        msg = model_print(lsvc_result, "LinearSVC - Train")
        st.write(msg)
        msg = model_print(adab_result, "AdaBoost - Train")
        st.write(msg)
        # 训练画图
        fig_train = plt_roc_auc([
            (lsvc_result, 'LinearSVC',),
            (adab_result, 'AdaBoost',),
        ], 'Train ROC')
        st.pyplot(fig_train)
    # 模型测试、显示结果
    with st.spinner("Testing, please wait..."):
        lsvc_test_result = model_score(lsvc, X_test, y_test)
        adab_test_result = model_score(adab, X_test, y_test)
    with col2:
        st.text("Testing Result")
        msg = model_print(lsvc_test_result, "LinearSVC - Test")
        st.write(msg)
        msg = model_print(adab_test_result, "AdaBoost - Test")
        st.write(msg)
        # 测试画图
        fig_test = plt_roc_auc([
            (lsvc_test_result, 'LinearSVC',),
            (adab_test_result, 'AdaBoost',),
        ], 'Validate ROC')
        st.pyplot(fig_test)


# 对生成的预测数据进行处理
def do_predict():
    global vars
    global lsvc, adab

    # 处理生成的预测数据的输入
    pocd_predict = pd.DataFrame(data=[vars], columns=COL_INPUT)
    pocd_predict = do_base_preprocessing(pocd_predict, with_y=False)
    st.text("Preview for detail of this predict data")
    st.write(pocd_predict)
    pocd_predict = do_predict_preprocessing(pocd_predict)

    # 进行预测并输出
    # LinearSVC
    pr = lsvc.predict(pocd_predict)
    pr = pr.astype(np.int)
    st.markdown(r"$\color{red}{LinearSVC}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(
        COL_Y[0]) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")
    # AdaBoost
    pr = adab.predict(pocd_predict)
    pr = pr.astype(np.int)
    st.markdown(r"$\color{red}{AdaBoost}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(
        COL_Y[0]) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")


if __name__ == "__main__":
    setup_uploader()
