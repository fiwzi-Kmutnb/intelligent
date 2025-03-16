import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, mean_absolute_error, r2_score


st.set_page_config(page_title="ML & Neural Network Analysis", layout="wide")
menu = {
    "Machine Learning ": "ml_explanation",
    "Machine Learning Model": "ml_analysis",
    "Neural Network": "nn_explanation",
    "Neural Network Model": "nn_image_analysis"
}

# ✅ ใช้ session_state เพื่อติดตามหน้าที่เลือก
if "page" not in st.session_state:
    st.session_state["page"] = list(menu.values())[0]  # หน้าแรกเป็น default

st.sidebar.write("### เลือกหน้า")

# ✅ สร้างปุ่มให้เปลี่ยนหน้า
for label, page in menu.items():
    if st.sidebar.button(label):
        st.session_state["page"] = page
        
if st.session_state["page"] == "ml_explanation":
    st.title("📘 Machine Learning คืออะไร?")
    st.write("""
             ## วิเคราห์การเงินโลก ในทุกๆประเทศ
                ได้มีโอกาศทำเกี่ยวกับคำนวณ GDP และการวิเคราะห์ข้อมูลทางการเงินของประเทศต่างๆ ในโลก โดยใช้ Machine Learning ในการทำนายข้อมูล และวิเคราะห์ข้อมูลทางการเงินของประเทศต่างๆ ในโลก 
             """)
    st.write("""
    ## การเตรียม Dataset
    ผมได้มีการหาข้อมูลของ Dataset มาจาก World Bank ซึ่งเป็นข้อมูลเกี่ยวกับประเทศต่าง ๆ ในโลก โดยมีข้อมูลที่สำคัญเช่น GDP, Inflation Rate, GNI per capita, Unemployment Rate และ Population
    link ของ Dataset: [link](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392)
    
    ## Data Features
    - **Country**: ชื่อประเทศ
    - **GDP (USD)**: Gross Domestic Product (GDP) ของประเทศ
    - **Year**: ปีที่เก็บข้อมูล
    - **Inflation Rate (%)**: อัตราการเงินเฟ้อ
    - **GNI per capita (USD)**: Gross National Income (GNI) ต่อคน
    - **Unemployment Rate (%)**: อัตราการว่างงาน
    - **Population**: จำนวนประชากร
    
    ## Model ที่ใช้งาน
    - **Classification**: ใช้ Decision Tree และ Random Forest
    - **Regression**: ใช้ Decision Tree และ Random Forest
    
    ## Classification Decision Tree
    (ต้นไม้ตัดสินใจสำหรับการจำแนกประเภท)
    เป็นโมเดลที่ใช้ในการจำแนกประเภทของข้อมูล โดยอาศัยโครงสร้างต้นไม้ที่มีกฎเกณฑ์ในการแบ่งข้อมูลออกเป็นกลุ่มย่อยต่างๆ จนได้คำตอบสุดท้าย

    - ใช้วิธีการแบ่งข้อมูลซ้ำๆ ตามเงื่อนไขที่เหมาะสมที่สุด (เช่น Gini Index หรือ Entropy)
    - เข้าใจง่ายและสามารถแสดงผลเป็นแผนภาพต้นไม้ได้
    - เหมาะกับข้อมูลที่มีโครงสร้างชัดเจน แต่มีโอกาสเกิด Overfitting ได้หากต้นไม้ลึกเกินไป
    
    ## Classification Random Forest
    (ป่าตัดสินใจสำหรับการจำแนกประเภท)
    เป็นเทคนิคที่ใช้ หลายๆ ต้นไม้ตัดสินใจ (Decision Trees) มาช่วยกันทำนายผล โดยรวมผลจากต้นไม้หลายต้นเพื่อลดข้อผิดพลาด

    - ใช้กระบวนการ Bootstrap Aggregation (Bagging) ในการสร้างต้นไม้หลายต้นจากชุดข้อมูลที่สุ่มขึ้นมา
    - ลดปัญหา Overfitting ที่มักเกิดกับ Decision Tree ได้ดีขึ้น
    - มีความแม่นยำสูงกว่าต้นไม้ตัดสินใจเพียงต้นเดียว แต่ตีความผลลัพธ์ได้ยากกว่า
    
    ## Regression Decision Tree
    (ต้นไม้ตัดสินใจสำหรับการพยากรณ์ค่าต่อเนื่อง)
    เป็นโมเดลที่ใช้ในการพยากรณ์ค่าตัวเลข โดยใช้โครงสร้างต้นไม้ตัดสินใจในการแบ่งกลุ่มข้อมูลตามเงื่อนไขที่เหมาะสม

    - ใช้วิธีลดค่า Mean Squared Error (MSE) หรือ Variance Reduction ในการเลือกเงื่อนไขที่เหมาะสม
    - สามารถอธิบายได้ง่าย แต่มีความเสี่ยงต่อ Overfitting หากต้นไม้ซับซ้อนเกินไป
    - เหมาะกับปัญหาการพยากรณ์เชิงตัวเลข เช่น คาดการณ์ราคา, คาดการณ์อุณหภูมิ เป็นต้น
    
    ## Regression Random Forest
    (ป่าตัดสินใจสำหรับการพยากรณ์ค่าต่อเนื่อง)
    เป็นการนำต้นไม้ตัดสินใจหลายต้นมาช่วยกันพยากรณ์ค่าโดยเฉลี่ยจากผลลัพธ์ของต้นไม้แต่ละต้น

    - ใช้แนวคิด Bagging ในการสุ่มตัวอย่างข้อมูลและสร้างต้นไม้หลายต้น
    - ลดความผันผวนของผลลัพธ์และลด Overfitting ได้ดี
    - เหมาะกับปัญหาที่ต้องการค่าพยากรณ์ที่แม่นยำ เช่น การทำนายยอดขาย, คาดการณ์แนวโน้มตลาด เป็นต้น
    
    
    ## ขั้นตอนการวิเคราะห์ข้อมูล
    ผมได้ทำการ import ไฟล์ csv และได้มีการกำจัด missing values โดยใช้ค่ามัธยฐานของข้อมูลในแต่ละประเทศที่มีข้อมูลที่ขาดหายมากกว่า 50% และเลือกเฉพาะคอลัมน์ที่สนใจเพื่อใช้ในการวิเคราะห์
    ```python
            file_path = "./dataset/world_bank_dataset.csv"
            df = pd.read_csv(file_path)

            threshold = len(df.columns) // 2
            df = df.dropna(thresh=threshold)
            

            selected_cols = ["Country", "GDP (USD)", "Year", "Inflation Rate (%)", "GNI per capita (USD)", "Unemployment Rate (%)", "Population"]
            df = df[selected_cols]
            for col in selected_cols[1:]: 
                df[col] = df[col].fillna(df[col].median())
    ```
    
    ## ในขั้นตอนต่อมา
    ผมได้ทำการแบ่งข้อมูลเป็น train และ test set และเลือกโมเดลที่ใช้ในการวิเคราะห์ข้อมูล โดยใช้ Decision Tree และ Random Forest ในการทำ Classification
    ซึ่งในขั้นตอนต่อมาจะเป็นการ เทรนในหมวด Classification มี 2 โมเดล คือ Decision Tree และ Random Forest
    
    ## การเทรน Decision Tree และ Random Forest ในหมวด Classification
    
    Decision Tree ในหมวดของ Classification 
    - สร้างคอลัมน์ใหม่ชื่อ "GDP_Class" เพื่อใช้เป็น label สำหรับการจำแนกประเภท
    - ใช้ค่ามัธยฐาน (median) ของ GDP (USD) เป็นเกณฑ์ในการแบ่งประเทศออกเป็น 2 กลุ่ม
        - ค่า GDP สูงกว่ามัธยฐาน → 1 (กลุ่มที่มี GDP สูง)
        - ค่า GDP ต่ำกว่าหรือเท่ากับมัธยฐาน → 0 (กลุ่มที่มี GDP ต่ำ)
    - astype(int) ใช้แปลงค่าผลลัพธ์จาก True/False เป็น 1/0 เพื่อให้ใช้งานได้กับโมเดล
    - X (Feature) → ข้อมูลที่ใช้เป็นปัจจัยในการทำนาย (ยกเว้นคอลัมน์ "Country", "GDP (USD)", และ "GDP_Class")
    - y (Target/Label) → ค่าเป้าหมายที่ต้องการทำนาย คือ "GDP_Class"
    - train_test_split() ใช้แบ่งข้อมูลเป็น 2 ส่วน
    - X_train, y_train → ใช้สำหรับฝึกโมเดล (80% ของข้อมูล)
    - X_test, y_test → ใช้สำหรับทดสอบโมเดล (20% ของข้อมูล)
    - random_state=42 ใช้กำหนดค่า seed เพื่อให้ได้ผลลัพธ์เดิมทุกครั้งที่รัน
    ```python
    df["GDP_Class"] = (df["GDP (USD)"] > df["GDP (USD)"].median()).astype(int)

        X = df.drop(columns=["Country", "GDP (USD)", "GDP_Class"])
        y = df["GDP_Class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_option == "Decision Tree":
            max_depth = st.slider("เลือกความลึกของต้นไม้ (max_depth)", 1, 10, 3)
            model = DecisionTreeClassifier()
    ```
    
    **Random Forest ในหมวดของ Classification**
    
     RandomForestClassifier(n_estimators=n_estimators) เป็นโมเดล Random Forest ที่ใช้ ต้นไม้ตัดสินใจ (Decision Trees) หลายต้น มาทำงานร่วมกันเพื่อช่วยจำแนกประเภทของข้อมูล

    - แต่ละต้นไม้ถูกสร้างขึ้นจาก ข้อมูลที่สุ่มมา
    - เมื่อต้องการทำนายผล ระบบจะให้ต้นไม้ทุกต้นช่วยกันโหวต และเลือกคำตอบที่มีเสียงข้างมาก
    - การใช้ต้นไม้หลายต้นช่วยให้โมเดล แม่นยำขึ้น และลดปัญหา Overfitting
    
    🔹 n_estimators คือจำนวนต้นไม้
    - ถ้าน้อยเกินไป → โมเดลอาจไม่เสถียร
    - ถ้ามากเกินไป → แม่นยำขึ้น แต่ใช้เวลาคำนวณมากขึ้น
    - ค่าเหมาะสมอยู่ที่ 100-200 ต้น ในการใช้งานทั่วไป
    
    ✅ เหมาะสำหรับงานที่ต้องการความแม่นยำสูง และต้องการลดข้อผิดพลาดของ Decision Tree แบบเดี่ยว
    
    ```python
            model = RandomForestClassifier(n_estimators=n_estimators)
    ```
    
    การเทรน Model ในหมวดของ Classification ของ Model ทั้ง 2 โมเดล คือ Decision Tree และ Random Forest
    
    **Performance Metrics**
    
    - Accuracy 🎯	ความถูกต้องโดยรวมของโมเดล
    - Precision ⚡	โมเดลทำนายค่าที่ถูกต้องในคลาสเป้าหมายได้แม่นยำแค่ไหน
    - Recall 🔄	โมเดลสามารถดึงข้อมูลที่เป็นกลุ่มเป้าหมายออกมาได้ดีแค่ไหน
    - F1-score ⭐	ค่าเฉลี่ยของ Precision และ Recall (เหมาะกับข้อมูลที่ไม่สมดุล)
    - Log Loss 📉	ใช้วัดค่าความผิดพลาดของโมเดล (ค่าต่ำกว่าดี)
    
    **การทำงานของกราฟ:**
    
    - ใช้ Matplotlib วาดกราฟแท่งเปรียบเทียบค่า Accuracy, Precision, Recall และ F1-score
    - กำหนด แกน y (0 - 1) เพราะค่าของ Metrics อยู่ในช่วงนี้
    
    ```python
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred)

        st.write("### 📈 ผลลัพธ์ของโมเดล Classification")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🎯 Accuracy", f"{acc:.2f}")
        col2.metric("⚡ Precision", f"{prec:.2f}")
        col3.metric("🔄 Recall", f"{recall:.2f}")
        col4.metric("⭐ F1-score", f"{f1:.2f}")
        col5.metric("📉 Log Loss", f"{loss:.2f}")

        fig, ax = plt.subplots()
        ax.bar(["Accuracy", "Precision", "Recall", "F1-score"], [acc, prec, recall, f1])
        ax.set_ylim(0, 1)
        ax.set_title("Classification Performance Metrics")
        st.pyplot(fig)
    ```
    
    ## การเทรน Decision Tree และ Random Forest ในหมวด Regression
    
    **Decision Tree ในหมวดของ Regression**
    
   - X (Feature) → ตัวแปรที่ใช้เป็นปัจจัยในการพยากรณ์ (ยกเว้น "Country", "GDP (USD)", "GNI per capita (USD)")
    - y (Target/Label) → ค่า GDP (USD) ที่ต้องการพยากรณ์
    
    **🔹 เหตุผลที่ตัด GDP (USD) ออกจาก X**
    - เพราะ GDP (USD) เป็นค่าที่เราต้องการพยากรณ์ หากใส่ไว้ใน X จะทำให้โมเดลมองเห็นคำตอบล่วงหน้า
    
    
    - แบ่งข้อมูลเป็น 80% สำหรับ Train และ 20% สำหรับ Test
    - random_state=42 → กำหนดค่า Seed เพื่อให้การสุ่มเหมือนเดิมทุกครั้งที่รัน
    ```python
     X = df.drop(columns=["Country", "GDP (USD)", "GNI per capita (USD)"])
        y = df["GDP (USD)"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

        if model_option == "Decision Tree":
            max_depth = st.slider("เลือกความลึกของต้นไม้ (max_depth)", 5, 100, 5)
            model = DecisionTreeRegressor(max_depth=max_depth)
    ```
    
    **Random Forest ในหมวดของ Regression**
    
    RandomForestRegressor(n_estimators=n_estimators, random_state=42) เป็นโมเดล Random Forest สำหรับการพยากรณ์ค่า (Regression) ซึ่งใช้ ต้นไม้ตัดสินใจ (Decision Trees) หลายต้น มาทำงานร่วมกันเพื่อพยากรณ์ค่าตัวเลข

    🔹 หลักการทำงาน

    - สร้างต้นไม้หลายต้น (Decision Trees) โดยใช้ชุดข้อมูลที่สุ่มมา
    - แต่ละต้นไม้จะทำนายค่าแยกกัน
    - นำค่าที่ได้จากต้นไม้ทุกต้นมาหาค่าเฉลี่ย เพื่อให้ได้คำตอบที่แม่นยำขึ้น
    
    🔹 n_estimators คือจำนวนต้นไม้
    - ถ้ามีต้นไม้ น้อยเกินไป → ผลลัพธ์อาจไม่นิ่ง (ค่าที่ได้อาจผันผวนสูง)
    - ถ้ามีต้นไม้ มากเกินไป → แม่นยำขึ้น แต่ใช้เวลาคำนวณมากขึ้น
    - ค่าที่นิยมใช้คือ 100-200 ต้น
    🔹 random_state=42 ทำให้ได้ผลลัพธ์เดิมทุกครั้งที่รัน

    ✅ เหมาะสำหรับงานที่ต้องการพยากรณ์ค่าตัวเลข เช่น การพยากรณ์ยอดขาย, ราคาบ้าน, หรือแนวโน้มเศรษฐกิจ 🚀

    ```python
            model = RandomForestClassifier(n_estimators=n_estimators)
    ```
    
    การเทรน Model ในหมวดของ Regression ของ Model ทั้ง 2 โมเดล คือ Decision Tree และ Random Forest
    
    **Performance Metrics**
    
    - MAE (Mean Absolute Error) 📉	ค่าความคลาดเคลื่อนเฉลี่ยระหว่างค่าจริง (y_test) กับค่าที่โมเดลพยากรณ์ (y_pred) ยิ่งต่ำยิ่งดี
    - R² Score 📊	ค่าที่บอกว่าโมเดลสามารถอธิบายความสัมพันธ์ของข้อมูลได้ดีแค่ไหน ค่ายิ่งใกล้ 1 ยิ่งดี
    
    **การทำงานของกราฟ:**

    - ใช้ scatter plot เพื่อแสดงค่าจริง (y_test) เทียบกับค่าพยากรณ์ (y_pred)
    - เส้นประ (r--) แสดงเส้นอ้างอิง (เส้นตรงถ้าโมเดลพยากรณ์ได้สมบูรณ์แบบ ทุกจุดควรอยู่บนเส้นนี้)
    - ถ้าจุดกระจายตัวใกล้เส้น → โมเดลแม่นยำ
    - ถ้าจุดกระจายตัวห่างจากเส้น → โมเดลมีข้อผิดพลาดมาก

    ```python
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### 📊 ผลลัพธ์ของโมเดล Regression")
        col1, col2 = st.columns(2)
        col1.metric("📉 MAE", f"{mae:.2f}")
        col2.metric("📊 R² Score", f"{r2:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, label="Actual vs Predicted")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # เส้นอ้างอิง
        ax.set_xlabel("Actual GDP (USD)")
        ax.set_ylabel("Predicted GDP (USD)")
        ax.set_title("Regression Prediction Performance")
        ax.legend()
        st.pyplot(fig)
    ```

    
    """)

elif st.session_state["page"] == "ml_analysis":
    file_path = "./dataset/world_bank_dataset.csv"
    df = pd.read_csv(file_path)

    threshold = len(df.columns) // 2
    df = df.dropna(thresh=threshold)
    

    selected_cols = ["Country", "GDP (USD)", "Year", "Inflation Rate (%)", "GNI per capita (USD)", "Unemployment Rate (%)", "Population"]
    df = df[selected_cols]
    for col in selected_cols[1:]: 
        df[col] = df[col].fillna(df[col].median())
    st.title("📊 การวิเคราะห์ข้อมูลการเงินโลกโดย Machine Learning")
    st.write("### 🔹 ข้อมูลปัจจุบัน")
    st.dataframe(df)
    problem_type = st.radio("เลือกประเภทของโมเดล", ["Classification", "Regression"])
    if problem_type == "Classification":
        df["GDP_Class"] = (df["GDP (USD)"] > df["GDP (USD)"].median()).astype(int)

        X = df.drop(columns=["Country", "GDP (USD)", "GDP_Class"])
        y = df["GDP_Class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_option = st.selectbox("เลือกโมเดล Classification", ["Decision Tree", "Random Forest"])

        if model_option == "Decision Tree":
            max_depth = st.slider("เลือกความลึกของต้นไม้ (max_depth)", 1, 10, 3)
            model = DecisionTreeClassifier()
        else:
            n_estimators = st.slider("เลือกจำนวนต้นไม้ (n_estimators)", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred)

        st.write("### 📈 ผลลัพธ์ของโมเดล Classification")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🎯 Accuracy", f"{acc:.2f}")
        col2.metric("⚡ Precision", f"{prec:.2f}")
        col3.metric("🔄 Recall", f"{recall:.2f}")
        col4.metric("⭐ F1-score", f"{f1:.2f}")
        col5.metric("📉 Log Loss", f"{loss:.2f}")

        fig, ax = plt.subplots()
        ax.bar(["Accuracy", "Precision", "Recall", "F1-score"], [acc, prec, recall, f1])
        ax.set_ylim(0, 1)
        ax.set_title("Classification Performance Metrics")
        st.pyplot(fig)
        
        
    else:
        X = df.drop(columns=["Country", "GDP (USD)", "GNI per capita (USD)"])
        y = df["GDP (USD)"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_option = st.selectbox("เลือกโมเดล Regression", ["Decision Tree", "Random Forest"])

        if model_option == "Decision Tree":
            max_depth = st.slider("เลือกความลึกของต้นไม้ (max_depth)", 5, 100, 5)
            model = DecisionTreeRegressor(max_depth=max_depth)
        else:
            n_estimators = st.slider("เลือกจำนวนต้นไม้ (n_estimators)", 1, 20, 5)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### 📊 ผลลัพธ์ของโมเดล Regression")
        col1, col2 = st.columns(2)
        col1.metric("📉 MAE", f"{mae:.2f}")
        col2.metric("📊 R² Score", f"{r2:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, label="Actual vs Predicted")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # เส้นอ้างอิง
        ax.set_xlabel("Actual GDP (USD)")
        ax.set_ylabel("Predicted GDP (USD)")
        ax.set_title("Regression Prediction Performance")
        ax.legend()
        st.pyplot(fig)
elif st.session_state["page"] == "nn_explanation":
    st.title("แยกโบโอมใน Minecraft ด้วย Neural Network CNN")
    st.write("""
        ผมทำเกี่ยวกับการจำแนกรูปภาพ โดยใช้ Neural Network แบบ CNN (Convolutional Neural Network) ซึ่งเป็นโมเดลที่เหมาะกับงานที่เกี่ยวกับภาพมากที่สุด
        โดยใช้ข้อมูลจากเกม Minecraft ที่มีรูปภาพของโบโอมต่างๆ และทำการแยกโบโอมออกจากรูปภาพ 
        ว่าแต่ละรูปภาพมีโอกาสเป็นไบโอมอะไรบ้าง โดยใช้โมเดล Neural Network CNN ที่เรียนรู้จากข้อมูลภาพที่มีโบโอมแต่ละชนิด
        
        ### ที่มาของ Dataset
        ผมได้มีการหาข้อมูล Dataset รูปภาพต่างๆ ของเกม Minecraft ใน kaggle ซึ่งมีรูปภาพของโบโอมต่างๆ ที่มีชื่อเรียกแตกต่างกันและมีที่คล้ายๆกันเยอะมาก
        
        ที่มา : [link](https://www.kaggle.com/datasets/willowc/minecraft-biomes)
        
        ### Data Features
        ชื่อไอดีของโบโอมทั้งหมด
        
        biome_10, biome_24, biome_0, biome_46, biome_47, biome_44, biome_45, biome_42, biome_43, biome_40, biome_41, biome_7, biome_11, biome_16, biome_25, biome_26, biome_4, biome_18, biome_132, biome_27, biome_28, biome_155, biome_156, biome_29, biome_157, biome_193, biome_21, biome_22, biome_149, biome_23, biome_151, biome_48, biome_49, biome_5, biome_19, biome_133, biome_30, biome_31, biome_158, biome_32, biome_33, biome_160, biome_161, biome_14, biome_15, biome_6, biome_134, biome_191, biome_35, biome_36, biome_163, biome_164, biome_1, biome_129, biome_2, biome_17, biome_130, biome_12, biome_13, biome_140, biome_3, biome_34, biome_131, biome_162, biome_20, biome_37, biome_39, biome_167, biome_38, biome_166, biome_165, biome_186, biome_192, biome_185, biome_184, biome_182, biome_183, biome_189, biome_187, biome_188, biome_190, biome_8, biome_179, biome_180, biome_178, biome_181, biome_9

        ## **📌 อธิบายกระบวนการฝึกโมเดล (Training Process)**
        โค้ดนี้ใช้ Convolutional Neural Network (CNN) เพื่อฝึกโมเดลจำแนกรูปภาพ โดยมีโครงสร้างและกระบวนการทำงานดังนี้

        **1️⃣ การเตรียมข้อมูล (Dataset Preparation)**
        - โหลดชุดข้อมูลภาพฝึก (train_ds) และชุดข้อมูลตรวจสอบ (val_ds)
        - ใช้ image_dataset_from_directory() โหลดรูปภาพจากโฟลเดอร์ dataset/train และ dataset/validation
        - กำหนดขนาดของภาพ (IMG_SIZE = (128, 128)) และกำหนดขนาดของชุดข้อมูล (BATCH_SIZE = 128)
        ```python
        train_ds = tf.keras.utils.image_dataset_from_directory(
            "./dataset/train",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            "./dataset/validation",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        ```
        
        
       ** 2️⃣ การสร้างโมเดล (Model Architecture)**
        โมเดลนี้ใช้ CNN (Convolutional Neural Network) ที่มีโครงสร้างดังนี้:

        - Rescaling → แปลงค่าพิกเซลจาก [0, 255] เป็น [0, 1] เพื่อให้การเรียนรู้มีประสิทธิภาพขึ้น
        - Conv2D + ReLU Activation + BatchNormalization
            - ใช้ 3 ชั้น Conv2D เพื่อดึงลักษณะเด่นจากภาพ
            - ใช้ BatchNormalization() เพื่อลดปัญหา vanishing gradient
            - ใช้ MaxPooling2D() เพื่อลดขนาดของภาพและพารามิเตอร์
        - GlobalAveragePooling2D → แปลงข้อมูลให้อยู่ในรูปแบบที่เหมาะกับ Fully Connected Layer
        - Dense Layers
            - ชั้น Dense(128, activation='relu') สำหรับเรียนรู้คุณสมบัติ
            - ชั้น Dropout(0.5) ลดโอกาส overfitting
            - ชั้นสุดท้าย Dense(len(train_ds.class_names), activation='softmax') ใช้สำหรับจำแนกประเภท
        ```python
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(128, 128, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(train_ds.class_names), activation='softmax') 
        ])
        ```
        
        **3️⃣ การเสริมข้อมูล (Data Augmentation)**
        - ใช้ RandomFlip, RandomRotation, และ RandomZoom เพื่อเพิ่มความหลากหลายของข้อมูล
        - ช่วยให้โมเดลเรียนรู้และจำแนกภาพได้ดีขึ้น
        
        ```python
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])
        ```
        
        - นำ augmentation มาใช้กับชุดฝึก (train_ds)

        ```python
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
        ```
        
       ** 4️⃣ การกำหนด Optimizer และ Learning Rate Schedule**
        - ใช้ Adam Optimizer พร้อม Learning Rate 0.0005
        - ใช้ ReduceLROnPlateau ลดค่า learning rate อัตโนมัติหาก val_loss ไม่ลดลง
        
        ```python
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        )

        optimizer = Adam(learning_rate=0.0005) 
        ```
        
        **5️⃣ การคอมไพล์และฝึกโมเดล**
        - ใช้ฟังก์ชัน Sparse Categorical Crossentropy เป็น loss function
        - ใช้ Accuracy เป็น metric
        
        ```python
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
        - ฝึกโมเดลเป็นเวลา 30 epochs
        ```python
        history = model.fit(train_ds, validation_data=val_ds, epochs=30)
        ```
        **6️⃣ การบันทึกโมเดล**
        - โมเดลที่ฝึกเสร็จแล้วถูกบันทึกเป็นไฟล์ .h5 เพื่อใช้งานภายหลัง
        
        ```python
        model.save("ibome_model.h5")
        ```
        
        **7️⃣ การแสดงผลลัพธ์**
        ใช้ Matplotlib แสดงกราฟ Accuracy ของ Training และ Validation
        
        ```python
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        ```
        
        ## **📌 อธิบายกระบวนการทำงานของโค้ดแยกไบโอมใน Minecraft ด้วย CNN**
        โค้ดนี้เป็น Streamlit Web App สำหรับการ แยกประเภทไบโอม (Biomes) ในเกม Minecraft โดยใช้โมเดล Neural Network CNN เพื่อตรวจจับและทำนายไบโอมจากรูปภาพที่อัปโหลด

        **1️⃣ โหลดและแสดงภาพที่อัปโหลด**
        - ให้ผู้ใช้เลือกและอัปโหลดรูปภาพ `(st.file_uploader())`
        - เปิดและแสดงภาพ`(st.image())`
        
        ```python
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ", type=["jpg", "png", "jpeg", "webp"])
        

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
            
        ```
        **2️⃣ โหลดโมเดล Neural Network**
        - ใช้ st.cache_resource เพื่อโหลดโมเดลเพียงครั้งเดียว
        - โหลดโมเดลที่ถูกฝึกมาแล้ว (minecraft.h5)
        
        ```python
        @st.cache_resource
        def load_model():
            model = tf.keras.models.load_model("./dataset/minecraft.h5")
            return model

        model = load_model()
        ```
        **3️⃣ แปลงรูปภาพเพื่อให้โมเดลสามารถทำนายได้**
        - เปลี่ยนภาพเป็นอาเรย์ (tf.keras.preprocessing.image.img_to_array())
        - ขยายมิติภาพ เพื่อให้โมเดลรับค่าเป็น batch
        
        ```python
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)
        ```
        
        **4️⃣ ทำนายไบโอมด้วยโมเดล**
        - ใช้ โมเดลที่โหลดมา (model.predict()) ทำนายค่า
        - แปลงผลลัพธ์เป็น Softmax probabilities
        - ดึงค่า ไบโอมที่มีโอกาสสูงสุด
        
        ```python
        predictions = model.predict(image_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy() 

        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class] * 100
        ```
        **5️⃣ จัดกลุ่มไบโอม (Biome Grouping)**
        - กำหนด ชื่อไบโอมที่โมเดลสามารถจำแนกได้ (class_names)
        - จัดกลุ่มไบโอมเป็นหมวดหมู่ เช่น Ocean, River, Desert, Jungle, Mountain เป็นต้น
        
        ```python
        biome_groups = {
            "Ocean": ["biome_10", "biome_24", "biome_0", "biome_46", "biome_47", "biome_44", "biome_45", "biome_42", "biome_43", "biome_40", "biome_41"],
            "River": ["biome_7", "biome_11"],
            "Beach": ["biome_16", "biome_25", "biome_26"],
            "Forest": ["biome_4", "biome_18", "biome_132", "biome_27", "biome_28", "biome_155", "biome_156", "biome_29", "biome_157", "biome_193"],
            "Jungle": ["biome_21", "biome_22", "biome_149", "biome_23", "biome_151", "biome_48", "biome_49"],
            "Taiga": ["biome_5", "biome_19", "biome_133", "biome_30", "biome_31", "biome_158", "biome_32", "biome_33", "biome_160", "biome_161"],
            "Mushroom": ["biome_14", "biome_15"],
            "Swamp": ["biome_6", "biome_134", "biome_191"],
            "Savanna": ["biome_35", "biome_36", "biome_163", "biome_164"],
            "Plains": ["biome_1", "biome_129"],
            "Desert": ["biome_2", "biome_17", "biome_130"],
            "Snowy": ["biome_12", "biome_13", "biome_140"],
            "Windswept": ["biome_3", "biome_34", "biome_131", "biome_162", "biome_20"],
            "Badlands": ["biome_37", "biome_39", "biome_167", "biome_38", "biome_166", "biome_165"],
            "Mountain": ["biome_186", "biome_192", "biome_185", "biome_184", "biome_182", "biome_183", "biome_189"],
            "Caves": ["biome_187", "biome_188", "biome_190"],
            "Nether": ["biome_8", "biome_179", "biome_180", "biome_178", "biome_181"],
            "The End": ["biome_9"]
        }
        ```
        **6️⃣ วิเคราะห์โอกาสของแต่ละไบโอม**
        - รวมเปอร์เซ็นต์ความน่าจะเป็นของไบโอมในแต่ละกลุ่ม
        - คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของความน่าจะเป็น
        - จัดอันดับไบโอมตามความเป็นไปได้มากที่สุด
        ```python
        biome_sums = {group: 0 for group in biome_groups}
        biome_details = {group: [] for group in biome_groups}

        sorted_indices = np.argsort(probabilities)[::-1]

        for i in sorted_indices:
            biome_name = class_names[i]
            prob_percent = probabilities[i] * 100

            for group, biomes in biome_groups.items():
                if biome_name in biomes:
                    biome_sums[group] += prob_percent
                    biome_details[group].append(prob_percent)

        biome_stats = {
            group: {
                "Total": round(biome_sums[group], 3),
                "Mean": round(np.mean(values) + np.std(values), 3) if values else 0,
                "Standard Deviation": round(np.std(values), 3) if values else 0
            } for group, values in biome_details.items()
        }

        sorted_biome_stats = dict(sorted(biome_stats.items(), key=lambda x: x[1]["Mean"], reverse=True))
        ```
        **7️⃣ แสดงผลลัพธ์**
        - แสดงไบโอมที่มีโอกาสสูงสุด
        - ใช้ st.write() แสดงผลลัพธ์ใน Streamlit
        ```python
        output_lines = []
        for group, stats in sorted_biome_stats.items():
            mean_value = stats["Mean"]
            output_lines.append(f"{group} ประมาณ {mean_value:.3f}%")

        st.write("### 📊 ผลลัพธ์จาก Neural Network มีโอกาสเป็นไบโอม: ")
        for i in output_lines:
            st.write(i)
        
        ```
        
        
    """)

elif st.session_state["page"] == "nn_image_analysis":
    st.title("🖼️ การแยกโบโอมใน Minecraft ด้วย Neural Network CNN")
    st.write(""" 
             กราฟการเทรนโมเดล Neural Network CNN แยกโบโอมใน Minecraft โดยใช้ข้อมูลจาก Dataset ที่มีรูปภาพของโบโอมต่างๆ และทำการแยกโบโอมออกจากรูปภาพ
             """)
    st.image("./train.jpg")
    
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ", type=["jpg", "png", "jpeg","webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)
        @st.cache_resource
        def load_model():
            model = tf.keras.models.load_model("./dataset/minecraft.h5")
            return model
        model = load_model()
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0) 

        predictions = model.predict(image_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy() 

        predicted_class = np.argmax(probabilities)  
        confidence = probabilities[predicted_class] * 100 
        class_names = ['biome_1', 'biome_10', 'biome_11', 'biome_12', 'biome_129', 'biome_13', 'biome_130', 'biome_131', 'biome_132', 'biome_133', 'biome_156', 'biome_157', 'biome_158', 'biome_16', 'biome_162', 'biome_17', 'biome_18', 'biome_19', 'biome_2', 'biome_21', 'biome_22', 'biome_26', 'biome_27', 'biome_28', 'biome_29', 'biome_3', 'biome_30', 'biome_31', 'biome_32', 'biome_33', 'biome_34', 'biome_35', 'biome_36', 'biome_37', 'biome_38', 'biome_39', 'biome_4', 'biome_45', 'biome_5', 'biome_6', 'biome_7']

        biome_groups = {
            "Ocean": [
                "biome_10", "biome_24", "biome_0", "biome_46", "biome_47", 
                "biome_44", "biome_45", "biome_42", "biome_43", "biome_40", "biome_41"
            ],
            "River": ["biome_7", "biome_11"],
            "Beach": ["biome_16", "biome_25", "biome_26"],
            "Forest": [
                "biome_4", "biome_18", "biome_132", "biome_27", "biome_28", 
                "biome_155", "biome_156", "biome_29", "biome_157", "biome_193"
            ],
            "Jungle": [
                "biome_21", "biome_22", "biome_149", "biome_23", "biome_151", 
                "biome_48", "biome_49"
            ],
            "Taiga": [
                "biome_5", "biome_19", "biome_133", "biome_30", "biome_31", 
                "biome_158", "biome_32", "biome_33", "biome_160", "biome_161"
            ],
            "Mushroom": ["biome_14", "biome_15"],
            "Swamp": ["biome_6", "biome_134", "biome_191"],
            "Savanna": [
                "biome_35", "biome_36", "biome_163", "biome_164"
            ],
            "Plains": ["biome_1", "biome_129"],
            "Desert": ["biome_2", "biome_17", "biome_130"],
            "Snowy": ["biome_12", "biome_13", "biome_140"],
            "Windswept": [
                "biome_3", "biome_34", "biome_131", "biome_162", "biome_20"
            ],
            "Badlands": [
                "biome_37", "biome_39", "biome_167", "biome_38", "biome_166", "biome_165"
            ],
            "Mountain": [
                "biome_186", "biome_192", "biome_185", "biome_184", "biome_182", 
                "biome_183", "biome_189"
            ],
            "Caves": ["biome_187", "biome_188", "biome_190"],
            "Nether": ["biome_8", "biome_179", "biome_180", "biome_178", "biome_181"],
            "The End": ["biome_9"]
        }

        biome_sums = {group: 0 for group in biome_groups}
        biome_percentages = {group: [] for group in biome_groups}
        biome_details = {group: [] for group in biome_groups} 

        sorted_indices = np.argsort(probabilities)[::-1]

        for i in sorted_indices:
            biome_name = class_names[i]
            prob_percent = probabilities[i] * 100

            for group, biomes in biome_groups.items():
                if biome_name in biomes:
                    biome_sums[group] += prob_percent
                    biome_percentages[group].append(f"{biome_name}: {prob_percent:.3f}%")
                    biome_details[group].append(prob_percent) 
        biome_stats = {
            group: {
                "Total": round(biome_sums[group], 3),
                "Mean": round(np.mean(values) + np.std(values), 3) if values else 0,
                "Standard Deviation": round(np.std(values), 3) if values else 0
            } for group, values in biome_details.items()
        }
        sorted_biome_stats = dict(sorted(biome_stats.items(), key=lambda x: x[1]["Mean"], reverse=True))
        output_lines = []
        for group, stats in sorted_biome_stats.items():
            total = stats["Total"]
            mean_value = stats["Mean"]
            std_dev = stats["Standard Deviation"]
            output_lines.append(f"{group} ประมาณ {mean_value:.3f}%")

        st.write("### 📊 ผลลัพธ์จาก Neural Network มีโอกาสเป็นไบโอม: ")
        st.write("### เป็นไบโอมที่มีโอกาสสูงสุด")
        st.write(output_lines[0])
        st.write("### อื่นๆ")
        for i in output_lines[1:5]:
            st.write(i)
            

