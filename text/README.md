{\rtf1\ansi\ansicpg1252\cocoartf2759
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Bold;\f1\froman\fcharset0 Times-Roman;\f2\fmodern\fcharset0 Courier;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid1\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid101\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc0\levelnfcn0\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{decimal\}}{\leveltext\leveltemplateid201\'01\'00;}{\levelnumbers\'01;}\fi-360\li720\lin720 }{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{circle\}}{\leveltext\leveltemplateid202\'01\uc0\u9702 ;}{\levelnumbers;}\fi-360\li1440\lin1440 }{\listname ;}\listid3}
{\list\listtemplateid4\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{disc\}}{\leveltext\leveltemplateid301\'01\uc0\u8226 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid4}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}{\listoverride\listid4\listoverridecount0\ls4}}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sa321\partightenfactor0

\f0\b\fs48 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Text Modality - Crime Report NLP Pipeline\
\pard\pardeftab720\sa298\partightenfactor0

\fs36 \cf0 Overview\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 This folder contains the NLP pipeline developed for the text modality of the AI Incident Reporting System project. The goal of this component is to process unstructured social media crime reports and convert them into a structured format that can be integrated with the outputs from the audio, PDF, image, and video modalities.\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Dataset\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls1\ilvl0
\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Dataset:
\f1\b0  CrimeReport (Kaggle)\
\ls1\ilvl0
\f0\b \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Data Type:
\f1\b0  Social media posts and crime-related news reports\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Tools Used\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls2\ilvl0
\f1\b0\fs24 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Python\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 pandas\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 spaCy (
\f2\fs26 en_core_web_sm
\f1\fs24 )\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 HuggingFace Transformers\
\ls2\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Jupyter Notebook\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Features\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The pipeline performs the following tasks:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls3\ilvl0\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	1	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Reads raw crime-related text data.\
\ls3\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	2	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Cleans and preprocesses text by removing noise and normalizing content.\
\ls3\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	3	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Tokenizes text and removes common stop words.\
\ls3\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	4	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Uses spaCy to extract named entities such as people, organizations, locations, and dates.\
\ls3\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	5	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Uses HuggingFace models for sentiment analysis.\
\ls3\ilvl0\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	6	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Uses HuggingFace zero-shot classification to categorize incidents into:\
\pard\tx940\tx1440\pardeftab720\li1440\fi-1440\sa240\partightenfactor0
\ls3\ilvl1\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u9702 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Theft / Robbery\
\ls3\ilvl1\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u9702 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Assault / Violence\
\ls3\ilvl1\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u9702 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Fire / Arson\
\ls3\ilvl1\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u9702 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Traffic Accident\
\ls3\ilvl1\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u9702 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Public Disturbance\
\ls3\ilvl1\kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u9702 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Other\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls3\ilvl0\cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	7	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Exports the final structured dataset as 
\f2\fs26 Text.csv
\f1\fs24 .\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Output\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The final CSV contains the following columns:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\sa240\partightenfactor0
\ls4\ilvl0
\f2\fs26 \cf0 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Text_ID
\f1\fs24 \
\ls4\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Source
\f1\fs24 \
\ls4\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Raw_Text
\f1\fs24 \
\ls4\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Sentiment
\f1\fs24 \
\ls4\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Entities
\f1\fs24 \
\ls4\ilvl0
\f2\fs26 \kerning1\expnd0\expndtw0 \outl0\strokewidth0 {\listtext	\uc0\u8226 	}\expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 Topic
\f1\fs24 \
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 How to Run\
\pard\pardeftab720\sa280\partightenfactor0

\fs28 \cf0 1. Install the required packages\
\pard\pardeftab720\partightenfactor0

\f2\b0\fs26 \cf0 pip install -r requirements.txt\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 2. Download the spaCy language model\
\pard\pardeftab720\partightenfactor0

\f2\b0\fs26 \cf0 python -m spacy download en_core_web_sm\
\pard\pardeftab720\sa280\partightenfactor0

\f0\b\fs28 \cf0 3. Run the notebook\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 Open and run:\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 text_nlp_pipeline_requirements_student_style.ipynb\
\pard\pardeftab720\sa298\partightenfactor0

\f0\b\fs36 \cf0 Output File\
\pard\pardeftab720\sa240\partightenfactor0

\f1\b0\fs24 \cf0 The pipeline generates:\
\pard\pardeftab720\partightenfactor0

\f2\fs26 \cf0 Text.csv\
}