data/diamonds.csv:
	mkdir -p data
	cd data; curl -LO https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv


q1: data/diamonds.csv
	mkdir -p figs
	python -B src/q1.py

q2: data/diamonds.csv
	mkdir -p figs
	python -B src/q2.py

q3: data/diamonds.csv
	python -B src/q3.py

q4: data/diamonds.csv
	python -B src/q4.py

q5: data/diamonds.csv
	python -B src/q5.py

clean:
	rm -r data

