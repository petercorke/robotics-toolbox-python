FILES = utility.py \
	transform.py \
	Quaternion.py \
	jacobian.py \
	trajectory.py \
	kinematics.py \
	dynamics.py \
	manipulability.py \
	Link.py \
	Robot.py \
	puma560.py \
	puma560akb.py \
	stanford.py \
	twolink.py
PFILES = $(addprefix robot/, $(FILES))

VERSION = $(shell svn info -r HEAD | grep Revision | awk '{print $$2}')
PASSWORD = $(shell cat .password)

all:
	echo $(VERSION)

code:
	zip -r pytb-$(VERSION) robot demo test -x \*/.svn/\* -x \*~ -x \*.pyc
	echo Uploading code version $(VERSION)
	googlecode_upload.py \
		-s 'Python code (snapshot)' \
		-p robotics-toolbox-python \
		-u peter.i.corke \
		-l Type-Source,OpSys-All,Type-Archive \
		-P $(PASSWORD) \
		pytb-$(VERSION).zip

doc: 
	epydoc $(PFILES)
	zip -r pytb-doc-$(VERSION) html
	echo Uploading doc version $(VERSION)
	googlecode_upload.py \
		-s 'HTML documentation (snapshot)' \
		-p robotics-toolbox-python \
		-u peter.i.corke \
		-l Type-Docs,OpSys-All,Type-Archive \
		-P $(PASSWORD) \
		pytb-doc-$(VERSION).zip
