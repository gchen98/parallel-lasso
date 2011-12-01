use gaw17;
show tables;
create temporary table size_gene(gene_symbol varchar(100),size int,primary key(gene_symbol));
insert into size_gene select gene_symbol,count(*) from snp_info_orig group by gene_symbol order by count(*) desc;
create temporary table common_gene(gene_symbol varchar(100),primary key(gene_symbol));
insert ignore into common_gene select gene_symbol from snp_info_orig where maf>.4;
select a.* from snp_info_orig as a,common_gene as b,size_gene as c where b.gene_symbol=a.gene_symbol and c.gene_symbol=a.gene_symbol and c.size>10 and c.size<30 order by gene_symbol,maf desc into outfile '/export/home/garykche/analysis/temp/causal_candidates.txt';
