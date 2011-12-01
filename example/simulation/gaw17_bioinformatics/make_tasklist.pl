#!/usr/bin/perl -w
use strict;

my $index=0;
open(IN,"snplist")|| die "Can't open snplist file\n";
while(my $line=<IN>){
  chomp($line);
  print "$index\n";
  ++$index;
}
close(IN);
