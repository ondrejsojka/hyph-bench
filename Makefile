# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it ms nl pl pt ru tr
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc cssk/cshyphen is/hyphenation-is th/orchid
# parse Wiktionary dumps into wordlists
process_all: prepare
	$(foreach l,$(WIKT_LANGS),rm -f ./data/$(l)/wiktionary/*.wlh;)
	$(foreach l,$(WIKT_LANGS),python ./scripts/process_dump.py --lang $(l);)

# create translate files for Wiktionary wordlists (which are created at first)
translate_all: process_all translate_other
	$(foreach l,$(WIKT_LANGS),rm -f ./data/$(l)/wiktionary/$(l)_*.tra;)
	$(foreach l,$(WIKT_LANGS),python ./scripts/make_tr.py ./data/$(l)/wiktionary/*.wlh;)

translate_other:
	$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*.tra;)
	$(foreach d,$(OTHER_DATASETS),python ./scripts/make_tr.py ./data/$(d)/*.wlh;)

# extract data from compressed Wiktionary dump and prepare directory structure
prepare:
	unzip ./data/wikt_dump.zip -d ./data
	$(foreach l,$(WIKT_LANGS),mkdir -p ./data/$(l)/wiktionary;)
