WIKT_LANGS = cs de el es it ms nl ru tr
WIKT_EN_LANGS = pl pt

# Transform 'en' into 'data/en/wiktionary/en_wiktionary.jsonl'
WIKT_JSONL_FILES = $(foreach lang,$(WIKT_LANGS),data/$(lang)/wiktionary/$(lang)_wiktionary.jsonl) $(foreach lang,$(WIKT_EN_LANGS),data/$(lang)/wiktionary/$(lang)_enwiktionary.jsonl)
WIKT_WLHAMB_FILES = $(foreach lang,$(WIKT_LANGS),data/$(lang)/wiktionary/$(lang)_wiktionary.wlhamb) $(foreach lang,$(WIKT_EN_LANGS),data/$(lang)/wiktionary/$(lang)_enwiktionary.wlhamb)
WIKT_WLH_FILES = $(foreach lang,$(WIKT_LANGS),data/$(lang)/wiktionary/$(lang)_wiktionary.wlh) $(foreach lang,$(WIKT_EN_LANGS),data/$(lang)/wiktionary/$(lang)_enwiktionary.wlh)
WIKT_TR_FILES = $(foreach lang,$(WIKT_LANGS),data/$(lang)/wiktionary/$(lang)_wiktionary.tr) $(foreach lang,$(WIKT_EN_LANGS),data/$(lang)/wiktionary/$(lang)_enwiktionary.tr)

# non-Wiktionary datasets
OTHER_DATASETS = cs/cshyphen_cstenten/cs_cstenten cs/cshyphen_ujc/cs_ujc cssk/cshyphen/cssk_cshyphen is/hyphenation-is/is_hyphis th/orchid/th_orchid de/wortliste/de_wortliste uk/cshyphen/uk_wiktionary
OTHER_WLH_FILES = $(foreach path,$(OTHER_DATASETS),data/$(path).wlh)
OTHER_TR_FILES = $(foreach path,$(OTHER_DATASETS),data/$(path).tr)

# cross-validate all datasets
cross_validate_all: translate
	@$(foreach d,$(wildcard data/*/*),python ./scripts/train_test.py -t -v -n 10 -p ./profiles/base.in $(d);)
	@$(foreach d,$(wildcard data/*/*),python ./scripts/train_test.py -t -v -n 10 -p ./profiles/cshyphen.in $(d);)
	@$(foreach d,$(wildcard data/*/*),python ./scripts/train_test.py -t -v -n 10 -p ./profiles/wortliste.in $(d);)
	@$(foreach d,$(wildcard data/*/*),python ./scripts/train_test.py -t -v -n 10 -p ./profiles/wortliste8.in $(d);)

# get statistics of all datasets
stats_all_datasets: disambiguate
	@$(foreach d,$(wildcard data/*/*/*.wlh),python ./scripts/statistics.py -d -t $(d);)

prepare: $(WIKT_JSONL_FILES)
process: $(WIKT_WLHAMB_FILES)
# create translate files
translate: $(WIKT_TR_FILES) $(OTHER_TR_FILES)
# resolve data ambiguities and long words
disambiguate: $(WIKT_WLH_FILES) $(OTHER_WLH_FILES)

%.jsonl : wikt_dump.zip
	@if echo $(WIKT_JSONL_FILES) | grep -q $@; then \
		LANG_CODE=$$(echo $(notdir $<) | cut -c 1-2); \
		mkdir -p data/$$LANG_CODE/wiktionary; \
		unzip -n wikt_dump.zip $(notdir $@) -d $(dir $@); \
	fi

%.wlhamb : %.jsonl
	@if echo $(WIKT_JSONL_FILES) | grep -q $<; then \
		LANG_CODE=$$(echo $(notdir $<) | cut -c 1-2); \
		python ./scripts/process_dump.py --lang $$LANG_CODE; \
	fi

%.wlh : %.wlhamb
	python ./scripts/disambiguate.py -v $<

%.tr : %.wlh
	python ./scripts/make_tr.py $<

.PHONY : clean
clean:
	rm -rf data/*/*/*.jsonl data/*/wiktionary/*.wlhamb data/*/*/*.wlh data/*/*/*.tr logs/
