#!/usr/bin/perl -w
use strict;

my %genes=();
open(IN,"positives")|| die "Can't open true positives file\n";
while(my $line=<IN>){
  chomp($line);
  my ($gene,$snp,$maf,$beta)=split(/\t/,$line);
  $genes{$snp}=$gene;
}
close(IN);
my $index=0;
open(IN,"snplist")|| die "Can't open snplist file\n";
while(my $line=<IN>){
  chomp($line);
  my $gene=$genes{$line};
  if (defined($gene)){
    print "$index\t$gene\n";
  }
  ++$index;
}
close(IN);
