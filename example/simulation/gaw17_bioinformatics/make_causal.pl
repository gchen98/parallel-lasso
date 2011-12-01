#!/usr/bin/perl -w
use strict;

my %chrs=();
my %positions=();
my %genes=();
my $beta=2;

my $total_genes=10;
my $total_genic_snps=10;
open(IN,"causal_candidates.txt")||die "Can't open causal genes file\n";
my %causal_genes=();
my %gene_counts=();
my $lastgene="";
my $snpcount=-1;
my $genecount=-1;
while(my $line=<IN>){
  chomp($line);
  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
  if ($lastgene ne $gene){
    $snpcount=0;
    ++$genecount;
  }else{
    ++$snpcount;
  }
  if ($genecount<$total_genes && $snpcount<$total_genic_snps){
    print "$line\n";
  }
  #$causal_genes{$gene}=$direction*$beta;
  $lastgene = $gene;
}
close(IN);


#open(IN,"snp_info")|| die "Can't open snpinfo file\n";
#<IN>;
#while(my $line=<IN>){
#  chomp($line);
#  my ($snp,$chr,$pos,$gene)=split(/\t/,$line);
#  my $effect=$causal_genes{$gene};
#  my $prefix=(defined($effect))?"CAUSAL":"NULL";
#  print "$chr\t".$prefix."_".$gene."_".$snp."\t0\t$pos\tA\tC\n";
#}
#
#my %effects=();
#open(IN,"positives")|| die "Can't open true positives file\n";
#while(my $line=<IN>){
#  chomp($line);
#  my ($gene,$snp,$maf,$beta)=split(/\t/,$line);
#  $effects{$snp}=$beta;
#}
#close(IN);
#my $index=0;
#while(my $snp=<STDIN>){
#  chomp($snp);
#  my $chr=$chrs{$snp};
#  my $gene=$genes{$snp};
#  my $pos=$positions{$snp};
#  my $effect=$effects{$snp};
#  my $prefix=(defined($effect))?"CAUSAL":"NULL";
#  print "$chr\t".$prefix."_".$gene."_".$snp."\t0\t$pos\tA\tC\n";
#  ++$index;
#}
