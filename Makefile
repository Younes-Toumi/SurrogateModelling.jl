run-test:
	@echo "running julia test from Makefile..."
	@julia --project -e 'using Pkg; Pkg.test()'; \
