# TimeLy
<img width="1512" alt="Screenshot 2023-10-22 at 6 55 12 AM" src="https://github.com/ritessshhh/TimeLy/assets/81812754/2bc51d23-1341-4f30-bc01-93999aa55641">

## Inspiration
We started **TimeLy** because we noticed a lot of useful information was hidden in the feedback and grades of courses offered in **Stony Brook University**. This information was not being used to its full potential. We believed that if this data could be understood easily, it would help students pick the **right courses**, and teachers could make their classes even better. So, we decided to create a tool that could make sense of all this data quickly and easily.

## What it does

TimeLy is a chatbot which performs a multi-faceted analysis of course feedback and grading data to extract actionable insights that benefit students, educators, and academic administrators. It also implements GPT-3 to give users human like experience. 

**For Students:**
Course Insights: Students receive personalized recommendations, aiding them in selecting courses that align with their academic goals and learning preferences.
Difficulty Levels: TimeLy categorizes courses into varying levels of difficulty, offering students a clear perspective to make informed decisions.

**For Educators:**
Feedback Analysis: It systematically analyzes student feedback, transforming it into clear, actionable insights for course improvement.
Sentiment Scores: Educators gain insights into the emotional tone of feedback, allowing them to address specific areas of concern or enhancement.

**For Administrators:**
Data Overview: Academic administrators access a consolidated view of courses’ performance, sentiments, and difficulty levels, enabling strategic decision-making.
Real-Time Queries: The platform supports real-time queries, offering instant insights to optimize academic offerings and student experiences.

With the help of advanced algorithms, TimeLy processes and analyzes educational data, translating it into normalized scores that offer a comparative view of courses. The sentiment analysis feature delves into the emotional context of feedback, presenting a balanced view of positive and negative sentiments.

With the integration of machine learning and AI, the platform becomes interactive. Users can ask questions and receive real-time answers, thanks to the integration of OpenAI GPT-3.5 Turbo. Flask's web interface ensures the platform is accessible and user-friendly, making complex data understandable and usable for decision-making.

## How we built it

The initial phase involved the extraction of data from Excel sheets. We wrote a Python script leveraging the Pandas library, an open-source data analysis and manipulation tool, to process and organize vast datasets efficiently.

Our code is designed to automatically check for pre-processed data stored in a **Parquet file** (to make the processing more faster), a columnar storage file format that is highly optimized for use with data processing frameworks. If the processed data is unavailable, our script initiates the extraction, transformation, and loading **(ETL)** process on the raw data from the Excel file.

For **sentiment analysis**, we employed a specialized sentiment analysis pipeline with the help of huggingface. It’s capable of processing large volumes of textual feedback to derive sentiment scores, categorizing them into positive, negative, or neutral sentiments. We addressed the challenge of handling extensive text data by implementing a truncation mechanism, ensuring optimal performance without compromising the quality of insights.

To transition TimeLy into an interactive, user-friendly platform, we utilized **Flask**, a micro web framework in Python. Flask enabled us to build a web-based interface that is both intuitive and responsive with the help of **HTML**, **CSS** and **JavaScript**. Users can input their queries in **natural language**, and the system, also integrated with the **OpenAI GPT-3.5 Turbo model**, provides real-time, intelligent, and contextual responses aside from the course schedule part.

We also incorporated **Spacy**, a leading library in **NLP (Natural Language Processing)**, to parse and categorize user inputs effectively, ensuring each query yields the most accurate and relevant results. The integration of these advanced technologies transformed TimeLy into a robust, interactive, and highly intuitive educational data analysis platform.

## Dependencies
```bash
Flask==2.3.3
gunicorn==20.1.0
Werkzeug==2.3.7
openai
transformers
pandas
spacy
flask
flask_cors
aiohttp==3.8.2
yarl==1.8.1
frozenlist==1.3.1
```

## Installation
```bash
[git clone https://github.com/mohamzamir/TImeLy](https://github.com/mohamzamir/TimeLy.git)
```
```bash
python -m spacy download en_core_web_sm
```


## What's next for TimeLy: A Course Recommender Tool
<img width="1512" alt="Screenshot 2023-10-22 at 6 56 18 AM" src="https://github.com/ritessshhh/TimeLy/assets/81812754/e8bd366e-5c6b-42a9-b32d-b02a68f58799">

**Feature Expansion:** We plan to enhance TimeLy by adding more features, such as personalized course recommendations based on individual student’s academic history, learning preferences, and career aspirations.
**Data Sources:** We aim to integrate additional data sources to provide a more comprehensive view and richer insights into courses, instructors, and institutions.
**AI Integration:** We are exploring opportunities to further harness AI, enhancing the tool’s predictive analytics capabilities to forecast trends and offer future-focused insights.
**User Community:** Building a community where users can share their experiences, provide feedback, and contribute to the continuous improvement of TimeLy.

## License
```
Copyright [2023] [Amir Hamza and Ritesh Chavan]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
