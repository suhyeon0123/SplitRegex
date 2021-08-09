J. C. Davis et al., Why aren't regular expressions a lingua franca? an empirical study on the re-use and portability of regular expressions, In Proceedings of ESEC/FSE 2019, 2019, pp. 443--454. https://doi.org/10.1145/3338906.3338909

`% jq -Mc '.pattern | strings' source.json > practical_regexes.json`

Each line in `practical_regexes.json` is a regex in quoted JSON strings.
