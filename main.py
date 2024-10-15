from Models import *

md = Models()
md.get_models(limit=10000, full=True)
md.filter_date()
md.filter_empty()
