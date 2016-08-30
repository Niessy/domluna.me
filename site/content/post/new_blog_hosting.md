+++
date = "2015-07-05T01:50:39-04:00"
title = "New Blog Hosting on S3!"
tags = ["S3", "Hugo"]

+++

So I've moved my site/blog off DigitalOcean and Ghost. Nothing agaisnt either, just that I couldn't justify the $5/month for something I could for ~$0/month (S3 is a few cents). I'm now hosted on S3, powered by Hugo and DNSed/SSLed by Cloudflare.

The S3 part was more approachable than I thought. I'm not sure why, but sometimes when I read AWS docs I feel like I'm reading another language. Anyway, the [walkthrough](https://docs.aws.amazon.com/AmazonS3/latest/dev/website-hosting-custom-domain-walkthrough.html) was easy to follow. The gist is  you put your assets in a bucket and link up an index.html and 404.html page, known as the Index and Error documents.

For my setup it's a 2 step process:

1. Generate assets (including index.html and 404.html).
2. Ship it to the S3 bucket.

### Hugo and s3cmd

[Hugo](hugo.spf13.com) is a static site generator, similar to Jekyll and friends but much faster. I'm not going to cover Hugo in this post, just think of it as a black box generator and that it's awesome!

Hugo to generates all  assets in a **public** including the index.html and 404.html pages whenever you run the `hugo` command or `hugo -t=themename` if you're using a theme. Next we need to ship the **public** folder to S3. We do this via the `s3cmd` cli tool.

```sh
hugo -t=casper # generates our site into a bunch of assets in the public folder

cd public # assumes we're at the top level in the site directory
s3cmd sync --delete-removed -P . s3://$BUCKET_NAME
```

The [s3cmd](http://s3tools.org/s3cmd) command is straightforward. The `sync` part says to sync the S3 bucket with our files, `--delete-removed` is telling it to also delete files we no longer have locally. `-P` is telling S3 to make these files viewable to the public. Lastly `. s3://$BUCKET_NAME` is saying move all the files in this directory to the S3 bucket.

You could make this process into a git commit hook but since I'm not running this 100 times a day I made it a script and called it a day.

### Cloudflare

I'm also using Cloudflare for the DNS bit of this operation. I should mention Cloudflare does waaaaaaaaaaaaay more than just DNS, they have caching, firewalls, analytics, optimization, tons of security stuff, SSL and the list goes on.

Here's the DNS.

![DNS setup](/images/cloudflare_dns.png)

Enabling SSL is even easier and free, they got a cool new thing called [Universal SSL](https://www.cloudflare.com/ssl).

To summarize:

* **Flexibile SSL** - don't need certificate on server, encrypt client to server but not Cloudflare to server.
* **Full SSL** - need certificate on server, encrypt both client-server and server-Cloudflare.
* **Full SSL (strict)** - same as Full SSL but the certificate needs to signed by a trusted authority.

Just to clarify **enabling SSL is free** but depending on how you get your ceritificate, that might not be.

The last step is to make sure all connections are https by default (upgrade http://... to https://...). Navigate to the **Page Rules** section of the dashboard and setup a rule similar to this.

![HTTPS upgrade setup](/images/cloudflare_page_rules.png)

So essentially the whole process, now we can post stuff! Hopefully you've learned something along the way.
