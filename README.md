# TimeLy

## Inspiration
We started **TimeLy** because we noticed a lot of useful information was hidden in the feedback and grades of courses. This information was not being used to its full potential. We believed that if this data could be understood easily, it would help students pick the **right courses**, and teachers could make their classes even better. So, we decided to create a tool that could make sense of all this data quickly and easily.

## What it does

It performs a multi-faceted analysis of course feedback and grading data to extract actionable insights that benefit students, educators, and academic administrators.

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

For **sentiment analysis**, we employed a specialized sentiment analysis pipeline. It’s capable of processing large volumes of textual feedback to derive sentiment scores, categorizing them into positive, negative, or neutral sentiments. We addressed the challenge of handling extensive text data by implementing a truncation mechanism, ensuring optimal performance without compromising the quality of insights.

To transition TimeLy into an interactive, user-friendly platform, we utilized **Flask**, a micro web framework in Python. Flask enabled us to build a web-based interface that is both intuitive and responsive with the help of **HTML**, **CSS** and **JavaScript**. Users can input their queries in **natural language**, and the system, also integrated with the **OpenAI GPT-3.5 Turbo model**, provides real-time, intelligent, and contextual responses aside from the course schedule part.

We also incorporated **Spacy**, a leading library in **NLP (Natural Language Processing)**, to parse and categorize user inputs effectively, ensuring each query yields the most accurate and relevant results. The integration of these advanced technologies transformed TimeLy into a robust, interactive, and highly intuitive educational data analysis platform.

## Challenges we ran into

We did run into some challenges. One big challenge was getting and handling a lot of text data. We had to figure out a way to read and understand this data without taking too much time. Another challenge was making the tool user-friendly. We wanted to make sure anyone could use it without needing to know a lot about data or programming. Balancing between making the tool powerful and keeping it easy to use was tough, but we learned a lot from it.

## Accomplishments that we're proud of

We are particularly proud of how TimeLy has came out from a concept into a functional, interactive tool that stands at the confluence of education and technology. Even though it lacks a lot of things, we are proud of what we built. 

**Interactivity:** The seamless integration of OpenAI GPT-3.5 Turbo, enabling real-time user interactions and intelligent responses, is an achievement that elevates the user experience.
**Sentiment Analysis:** Implementing a robust sentiment analysis feature that provides nuanced insights into the emotional context of student feedback is another accomplishment.
**User Experience:** We successfully created an intuitive user interface using Flask, ensuring that complex data is accessible and understandable to all users, irrespective of their technical expertise.


## What we learned

**Technical Skills:** We honed our skills in Python, data analysis, and machine learning. Working with libraries like Pandas, Spacy, and integrating OpenAI was a rich learning experience.
**User Engagement:** We learned the pivotal role of user experience, driving us to make TimeLy as intuitive and user-friendly as possible while retaining its technical robustness.
**Data Insights:** The project deepened our understanding of the power of data and how processed, analyzed data can be a goldmine of insights for students, educators, and institutions.

## What's next for TimeLy: A Course Recommender Tool

**Feature Expansion:** We plan to enhance TimeLy by adding more features, such as personalized course recommendations based on individual student’s academic history, learning preferences, and career aspirations.
**Data Sources:** We aim to integrate additional data sources to provide a more comprehensive view and richer insights into courses, instructors, and institutions.
**AI Integration:** We are exploring opportunities to further harness AI, enhancing the tool’s predictive analytics capabilities to forecast trends and offer future-focused insights.
**User Community:** Building a community where users can share their experiences, provide feedback, and contribute to the continuous improvement of TimeLy.
