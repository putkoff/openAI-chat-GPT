import os
def changeGlob(x,y):
    globals()[x]=y
    return y
  
changeGlob('home',os.getcwd())
choosIt = ["chat","translate","qanda","parse","editcode","debugcode","convertcode","writecode"]
categories={"completions":["chat","translate","qanda","parse"],"coding":["editcode","debugcode","convertcode","writecode"],"embeddings":["text_search_doc","similarityIt","text_similarityIt","text_search_queryIt","text_embeddingIt","text_insertIt","text_editIt","search_documentIt","s","instructIt","code_editIt","code_search_codeIt","code_search_textIt"],"moderation":["moderate"],"images":["image_create","image_edit","image_variatoin"]}
parameters={'model':{'object':'str','scale':'inherit','default':'text-davinci-003'},
            'max_tokens':{'object':'int','scale':'range','range':{0:2048},'default':2000,},
            'logit_bias':{'object':'map','scale':'range','range':{-100:100},'default':''},
            'size':{'object':'str','default':'1024x1024','scale':'choice','choice':['256x256','512x512','1024x1024']},
            'temperature':{'object':'float','default':float(0.7),'scale':'range','range':{-2.0:2.0}},
            'best_of':{'object':'int','default':1,'scale':'range','range':{0:10}},
            'top_p':{'object':'float','default':float(0.0),'scale':'range','range':{0.0:1.0}},
            'frequency_penalty':{'object':'float','default':float(0.0),'scale':'range','range':{-2.0:2.0}},
            'presence_penalty':{'object':'float','default':float(0.0),'scale':'range','range':{-2.0:2.0}},
            'log_probs':{'object':'int','default':int(1),'scale':'range','range':{1:10}},
            'stop':{'object':'str','default':'','scale':'array','range':{0:4}},
            'echo':{'object':'bool','default':'False','scale':'choice','choice':['True','False']},
            'n':{'object':'int','default':int(1),'scale':'range','range':{1:10}},
            'stream':{'object':'bool','default':'False','scale':'choice','choice':['True','False']},
            'suffix':{'object':'str','default':'','scale':'range','range':{0:1}},
            'prompt':{'object':'str','default':'""','scale':'inherit'},
            'model':{'object':'str','default':'text-davinci-003','scale':'array','array':['completion','edit','code','embedding']},
            'input':{'object':'str','default':'""','scale':'inherit'},
            'instruction':{'object':'str','default':"''",'scale':'inherit'},
            'response_format':{'object':'str','default':"url",'scale':'choice','choice':['url','b64_json']},
            'image':{'object':'str','default':'""','scale':'upload','upload':{'type':['PNG','png'],'size':{'scale':{0:4},'allocation':'MB'}}},
            'mask':{'object':'str','default':'""','scale':'upload','upload':{'type':['PNG','png'],'size':{'scale':{0:4},'allocation':'MB'}}},
            'file':{'object':'str','default':'""','scale':'upload','upload':{'type':['jsonl'],'size':{'scale':{0:'inf'}},'allocation':'MB'}},
            'purpose':{'object':'str','default':'""','scale':'inherit'},'file_id':{'object':'str','default':'','scale':'inherit'},
            'user':{'object':'str','default':'defaultUser','scale':'inherit'}
            }
specifications={"completion":{'type':'completions','model':{'default':'text-davinci-003','choices':['text-ada-001','text-davinci-003','text-curie-001','text-babbage-001']},"clients":"@mulChoice(specifications[typ]['model']['choices'],'model')\n@mulChoice(['True','False'],'stream')",},
                "coding":{'type':'completions','model':{'default':'text-davinci-003','choices':['code-cushman-001','code-davinci-002']},"clients":"@mulChoice(['Python','Java','C++','JavaScript','Go','Julia','R','MATLAB','Swift','Prolog','Lisp','Haskell','Erlang','Scala','Clojure','F#','OCaml','Kotlin','Dart'],'language')\n@mulChoice(specifications[typ]['model']['choices'],'model')\n\n@mulChoice(['True','False'],'echo')\n@mulChoice(['True','False'],'stream')",},
                "embeddings":{'type':'embeddings','model':{'default':'text-embedding-ada-002','choices':['text-ada-001','text-davinci-003','text-curie-001','text-babbage-001']},"clients":"@mulChoice(specifications[typ]['model']['choices'],'model')"},
                "moderations":{'type':'moderation','model':{'default':'text-davinci-003','model':{'default':'text-davinci-003','choices':['text-davinci-003','text-moderation-001']},"clients":"@mulChoice(specifications[typ]['model']['choices'],'model')\n"}},
                "edits":{'type':'edits','model':{'default':'text-ada-001','choices':['text-ada-001','text-davinci-003','text-curie-001','text-babbage-001']},"clients":"@mulChoice(specifications[typ]['model']['choices'],'model')\n@mulChoice(['True','False'],'echo')\n@mulChoice(['True','False'],'stream')",},
                "images":{'type':'images','model':{},"clients":"\t@imageSize()\n@mulChoice(['url','b64_json'])"},
                "moderation":{'type':'moderate','model':{'default':"text-moderation-003",'choices':["text-moderation-003","text-moderation-001"]},"clients":"@mulChoice(specifications[typ]['model']['choices'],'model')\n\t"},                          
                "edit":{'type':'edit','delims':['',''],'model':{'default':"text-davinci-edit-001",'clients':["text-davinci-edit-001"]},"clients":"@mulChoice(specifications[typ]['model']['choices'],'model')\n\t"},

                "uploadfile":{'type':'completion','delims':['file','name']},
                
                "translate":{'type':'completion','delims':['#i will need for you to translate [text] into [languages]:','languages:','text:'],'vars':['languages','text']},
                "qanda":{'type':'completion','delims':['Q:','A:'],'vars':['question']},
                "chat":{'type':'completion','delims':['',''],'vars':['prompt']},
                "parse":{'type':'completion','delims':['#this query is for parsing, a [summary] of the [data] will be given in order to parse specific [variables]:','summary:','data:','variables:'],'vars':['summary','data','variables']},
                
                "writecode":{'type':'completion','delims':['#write code in [language] based off of specific [instruction]:','language:','instruction'],'vars':['language','instruction']},
                "editcode":{'type':'completion','delims':['#edit based off of specific [instructions] i will need you to write [code]:','instructon:','code:'],'vars':['instructon','code']},       
                "debugcode":{'type':'completion','delims':['#debug [code] based off of specific [instructions]:','code:','instructions:'],'vars':['code','instructions']},
                "convertcode":{'type':'completion','delims':['#convert [code] to [language]:','code:','language:'],'vars':['code','language']},

                "image_create":{'type':'images','delims':['','']},
                "image_edit":{'type':'images','delims':['','']},
                "image_variation":{'type':'images','delims':['','']},
                
                'moderate':{'delims':['moderate this text:','']},              
                "text_search_doc":{'type':'embedding','delims':['',''],},
                "similarity":{'type':'embedding','delims':['','']},
                "text_similarity":{'type':'embedding','delims':['','']},
                "text_search_query":{'type':'embedding','delims':['','']},
                "text_embedding":{'type':'embedding','delims':['','']},
                "text_insert":{'type':'embedding','delims':['','']},
                "text_edit":{'type':'embedding','delims':['','']},
                "search_document":{'type':'embedding','delims':['','']},
                "search_query":{'type':'embedding','delims':['', '']},
                "instruct":{'type':'embedding','delims':['','']},
                "code_edit":{'type':'embedding','delims':['','']},
                "code_search_code":{'type':'embedding','delims':['','']},
                "code_search_text":{'type':'embedding','delims':['','']}}
choi = ["translate","qanda","chat","parse","mention","writecode","uploadcode","editcode","debugcode","convertcode","image_create","image_edit","image_variation","text_search_doc","similarity","text_similarity","text_search_query","text_embedding","text_insert","text_edit","search_document","search_query","instruct","code_edit","code_search_code","code_search_text","moderation","edit","private","public","help","params",'uploadfile']            
descriptions= {"completions":'input what youd like to say to the bot, Have a chat with ChatGPT',#str(categories['completions']),
                "embeddings": str(categories['embeddings']),
                "completion":str(categories['completions']),
                "moderation":str(categories['moderation']),
                "images":str(categories['images']),
                "choices":"choose from the selection",
                "types":"choose from the selection",
                "coding":"write some code",
                'uploadcode':'[prompt]-describe what your focus is;[code]- upload your code',
                'temp':'pick the randomness of your interaction',
                'qanda':'[prompt]- input a question,question mark will auto add',
                'translate':'[prompt] - enter the text you would like to translate;[language] -enter the desired languages',
                'text_search_doc':'a',
                'writecode':'[prompt]-describe the code; [language] - specify the target language',
                'debugcode':'[prompt]-describe what your focus is;[code]- enter your code',
                'editcode':'[prompt]-describe what your focus is;[code]- enter your code',
                'convertcode':'[code]-input your code;[language]-input the language youd like to convert to',
                'image_variation':'[image]- upload an image of your choice; [prompt]- input how you would like it edited',
                'mention':'[prompt] - input what youd like to say to the bot',
                'chat':'[prompt] - input what youd like to say to the bot, Have a chat with ChatGPT',
                'image_create':'[prompt]- input what image you would like to have formulated',
                'public':'Toggle public access',
                'private':'Toggle private access',
                'help':'will display all descriptions',
                'temp':'Temperature will allow you to pick the randomness of your interaction; range(0:2) _ input(integer)',
                'parse':'[summerize]-summarize the text;[subjects]-comma seperated subjects to parse;[prompt]-entertext',
                'moderation':'[input] - input text you would like to have moderated',
                'edit':'[input]-enter your text; [instruction]- tell it what you want it to do.',
                'shouldBeAllGood':'below-----------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
                'stillInTesting':'below-----------VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV',
                'similarity':'where results are ranked by relevance to a query string',
                'text_similarity':'Captures semantic similarity between pieces of text.',
                'text_search_query':'Semantic information retrieval over documents.',
                'text_embedding':'Get a vector representation of a given input that can be easily consumed by machine learning model',
                'text_insert':'insert text',
                'text_edit':'edit text',
                'search_document':'where results are ranked by relevance to a document',
                'search_query':'search query ',
                'code_edit':'specify the revisions that you are looking to make in the code',
                'code_search_code':'Find relevant code with a query in natural language.',
                'code_search_text':'text search in code',
                'image_edit':'[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited',
                'params': 'lists definitions and information about all parameters',
                'uploadfile': 'upload a file to be used in future queries'}
paramDescs = {"choices":"choose from the selection",'whole': ['all','model', 'prompt', 'suffix', 'max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user', 'input', 'instruction', 'size', 'response_format', 'image', 'mask', 'file', 'purpose', 'file_id', 'training_file', 'validation_file', 'n_epochs', 'batch_size', 'learning_rate_multiplier', 'prompt_loss_weight', 'compute_classification_metrics', 'classification_n_classes', 'classification_positive_class', 'classification_betas', 'fine_tune_id', 'engine_id'],'all':{'model': 'The ID of the model to use for this request','prompt': 'The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.Note that &lt;|endoftext|&gt; is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.','suffix': 'The suffix that comes after a completion of inserted text.','max_tokens': 'The maximum number of tokens to generate in the completion.The token count of your prompt plus max_tokens cannot exceed the model context length. Most model have a context length of 2048 tokens (except for the newest model, which support 4096).','temperature': 'What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well: defined answer.We generally recommend altering this or top_p but not both.','top_p': 'An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.We generally recommend altering this or temperature but not both.','n': 'How many completions to generate for each prompt.Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop.','stream': 'Whether to stream back partial progress. If set, tokens will be sent as data: only server: sent events as they become available, with the stream terminated by a data: [DONE] message.','logprobs': 'Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.The maximum value for logprobs is 5. If you need more than this, please contact us through our Help center and describe your use case.','echo': 'Echo back the prompt in addition to the completion','stop': 'Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.','presence_penalty': 'Number between : 2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model likelihood to talk about new topics.See more information about frequency and presence penalties.','frequency_penalty': 'Number between : 2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model likelihood to repeat the same line verbatim.See more information about frequency and presence penalties.','best_of': 'Generates best_of completions server: side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop.','logit_bias': 'Modify the likelihood of specified tokens appearing in the completion.Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from : 100 to 100. You can use this tokenizer tool (which works for both GPT: 2 and GPT: 3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between : 1 and 1 should decrease or increase likelihood of selection; values like : 100 or 100 should result in a ban or exclusive selection of the relevant token.As an example, you can pass {"50256": : 100} to prevent the &lt;|endoftext|&gt; token from being generated.','user': 'A unique identifier representing your end: user, which can help OpenAI to monitor and detect abuse. Learn more.','input': 'The input text to use as a starting point for the edit.','instruction': 'The instruction that tells the model how to edit the prompt.','size': 'The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.','response_format': 'The format in which the generated images are returned. Must be one of url or .','image': 'The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask.','mask': 'An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image.','file': 'Name of the JSON Lines file to be uploaded.If the purpose is set to "fine: tune", each line is a JSON record with "prompt" and "completion" fields representing your training examples.','purpose': 'The intended purpose of the uploaded documents.Use "fine: tune" for Fine: tuning. This allows us to validate the format of the uploaded file.','file_id': 'The ID of the file to use for this request','training_file': 'The ID of an uploaded file that contains training data.See upload file for how to upload a file.Your dataset must be formatted as a JSONL file, where each trainingexample is a JSON object with the keys "prompt" and "completion".Additionally, you must upload your file with the purpose fine: tune.See the fine: tuning guide for more details.','validation_file': 'The ID of an uploaded file that contains validation data.If you provide this file, the data is used to generate validationmetrics periodically during fine: tuning. These metrics can be viewed inthe fine: tuning results file.Your train and validation data should be mutually exclusive.Your dataset must be formatted as a JSONL file, where each validationexample is a JSON object with the keys "prompt" and "completion".Additionally, you must upload your file with the purpose fine: tune.See the fine: tuning guide for more details.','n_epochs': 'The number of epochs to train the model for. An epoch refers to onefull cycle through the training dataset.','batch_size': 'The batch size to use for training. The batch size is the number oftraining examples used to train a single forward and backward pass.By default, the batch size will be dynamically configured to be~0.2% of the number of examples in the training set, capped at 256 : in general, weve found that larger batch sizes tend to work betterfor larger datasets.','learning_rate_multiplier': 'The learning rate multiplier to use for training.The fine: tuning learning rate is the original learning rate used forpretraining multiplied by this value.By default, the learning rate multiplier is the 0.05, 0.1, or 0.2depending on final batch_size (larger learning rates tend toperform better with larger batch sizes). We recommend experimentingwith values in the range 0.02 to 0.2 to see what produces the bestresults.','prompt_loss_weight': 'The weight to use for loss on the prompt tokens. This controls howmuch the model tries to learn to generate the prompt (as comparedto the completion which always has a weight of 1.0), and can adda stabilizing effect to training when completions are short.If prompts are extremely long (relative to completions), it may makesense to reduce this weight so as to avoid over: prioritizinglearning the prompt.','compute_classification_metrics': 'If set, we calculate classification: specific metrics such as accuracyand F: 1 score using the validation set at the end of every epoch.These metrics can be viewed in the results file.In order to compute classification metrics, you must provide avalidation_file. Additionally, you mustspecify classification_n_classes for multiclass classification orclassification_positive_class for binary classification.','classification_n_classes': 'The number of classes in a classification task.This parameter is required for multiclass classification.','classification_positive_class': 'The positive class in binary classification.This parameter is needed to generate precision, recall, and F1metrics when doing binary classification.','classification_betas': 'If this is provided, we calculate F: beta scores at the specifiedbeta values. The F: beta score is a generalization of F: 1 score.This is only used for binary classification.With a beta of 1 (i.e. the F: 1 score), precision and recall aregiven the same weight. A larger beta score puts more weight onrecall and less on precision. A smaller beta score puts more weighton precision and less on recall.','fine_tune_id': 'The ID of the fine: tune job','engine_id': 'The ID of the engine to use for this request'}}
