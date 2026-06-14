pyfiles = kinematics.py dynamics.py icra2021.py

all: $(pyfiles)

clean:
	rm *.py

$(pyfiles): %.py: %.ipynb
	jupyter nbconvert --to script $<
	ipython --matplotlib=osx $@
