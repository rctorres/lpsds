# fast-ds

## Helper functions for data science projects

Common tools for data science projects.

You must set the following environment variables:

## LPSDS_HANDLER

This variable defines which file handler to use. It can have 2 values:

* LocalFileHandler: for files located in your local machine.
* S3FileHandler: for files located in an S3 bucket. You must have your AWS access tokens properly set for that.

## LPSDS_BUCKET

A base directory (if using LocalFileHandler) or the S3 bucket (if using S3FileHandler) that will be used as base point for input and output operations. The value should not have "S3://" in case it is a S3 bucket. Just the bucket name will do.
