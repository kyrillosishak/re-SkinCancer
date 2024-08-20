MARKDOWNS := reproducingSkinCancer.md correctingSkinCancer.md exploreDuplicate.md 
NOTEBOOKS := notebooks/reproducingSkinCancer.ipynb notebooks/correctingSkinCancer.ipynb notebooks/exploreDuplicate.ipynb 

all: $(NOTEBOOKS)

clean:
	rm -f $(NOTEBOOKS)

notebooks/%.ipynb: markdowns/%.md markdowns/front-matter.md
	pandoc --resource-path=assets/ --embed-resources --standalone --wrap=none -i markdowns/front-matter.md $< -o $@