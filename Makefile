# languages provided in Wiktionary dump
WIKT_LANGS = cs de el es it ms nl pl pt ru tr

# non-Wiktionary datasets
# cssk/cshyphen is special as it is weighted
CSSK = cssk/cshyphen
OTHER_DATASETS = cs/cshyphen_cstenten cs/cshyphen_ujc is/hyphenation-is th/orchid de/wortliste uk/wiktionary

# cross-validate all datasets
cross_validate_all: translate_all
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/base.in $(d);)
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/cshyphen.in $(d);)
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/wortliste.in $(d);)
	@$(foreach d,$(wildcard data/*/*),uv run python -m scripts.train_test -t -v -n 10 -p ./profiles/wortliste8.in $(d);)

# get statistics of all datasets
stats_all_datasets: disambiguate_all
	@$(foreach d,$(wildcard data/*/*/*_dis.wlh),uv run python -m scripts.statistics -d -t $(d);)

# parse Wiktionary dumps into wordlists
process_wikt: prepare_wikt
	@$(foreach l,$(WIKT_LANGS),rm -f ./data/$(l)/wiktionary/*.wlh;)
	@$(foreach l,$(WIKT_LANGS),uv run python -m scripts.process_dump --lang $(l);)

# create translate files
translate_all: translate_wikt translate_other

translate_wikt: disambiguate_wikt
	@$(foreach l,$(wildcard data/*/wiktionary/*.tra),rm -f $(l);)
	@$(foreach l,$(wildcard data/*/wiktionary/*_dis.wlh),uv run python -m scripts.make_tr $(l);)

translate_other: disambiguate_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*.tra;)
	@$(foreach d,$(OTHER_DATASETS),uv run python -m scripts.make_tr ./data/$(d)/*_dis.wlh;)
	@rm -f ./data/$(CSSK)/*.tra
	@uv run python -m scripts.make_tr ./data/$(CSSK)/*_expanded.wlh


# resolve data ambiguities and long words
disambiguate_all: disambiguate_wikt disambiguate_other

disambiguate_wikt: process_wikt
	@$(foreach d,$(wildcard data/*/wiktionary/*_dis.wlh),rm -f $(d);)
	@$(foreach d,$(WIKT_LANGS),uv run python -m scripts.disambiguate ./data/$(d)/wiktionary/*.wlh;)

disambiguate_other: prepare_other
	@$(foreach d,$(OTHER_DATASETS),rm -f ./data/$(d)/*_dis.wlh;)
	@$(foreach d,$(OTHER_DATASETS),uv run python -m scripts.disambiguate ./data/$(d)/*.wlh;)

# extract data from compressed Wiktionary dump and prepare directory structure
prepare_wikt:
	@mkdir -p ./wikt_dump
	@unzip -o -d ./wikt_dump ./wikt_dump.zip
	@$(foreach l,$(WIKT_LANGS),mkdir -p ./data/$(l)/wiktionary;)

# expand weighted cssk/cshyphen dataset
prepare_other:
	@uv run python -m scripts.expand_weights ./data/$(CSSK)/*.wlhw

include thesis/thesis.mk