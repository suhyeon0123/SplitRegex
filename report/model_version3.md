## Model architecture
![ex_screenshot](./model_version3.JPG)
## Result

- unidirectional model

|   |string equal|dfa equal|membership equal|total|
|------|---|---|---|---|
|star0|0|0|-|2000|
|star1|0|0|-|1998|
|star2|0|0|-|1998|
|star3|0|0|-|1998|

- bidirectional model

|   |string equal|dfa equal|membership equal|total|
|------|---|---|---|---|
|star0|213|1|-|214/2000|
|star1|67|15|-|82/1998|
|star2|6|6|-|12/1998|
|star3|10|6|-|16/1998|

- bidirectional + attention(only positive samples)

|   |string equal|dfa equal|membership equal|total|
|------|---|---|---|---|
|star0|1017|3|-|1020/2000|
|star1|444|93|-|537/1998|
|star2|221|79|-|300/1998|
|star3|133|80|-|213/1998|

- bidirectional + attention(both positive and negative samples)

|   |string equal|dfa equal|membership equal|total|
|------|---|---|---|---|
|star0|994|1|-|995/2000|
|star1|418|84|-|502/1998|
|star2|224|78|-|302/1998|
|star3|139|62|-|201/1998|
