+++
title = "Setting up Ghost with Docker Machine"
date = "2015-03-05"
share = true
tags = ["Docker", "docker-machine", "Ghost"]
+++

This is my second go at using Ghost for blogging, it was my first choice a while back. I then sort of switched to Github Pages. This, however, was still a Ghost blog.

I used [Buster](https://github.com/axitkhurana/buster) to generate the static files from a running Ghost server. It turns out while it was nice having Github host the site the workflow was pretty awful and I didn't end up writing much.

So I took my hand at doing a site not Ghost related. This would give me more versatility than just blogging and I figured this would be good. Turns out all that extra versatility and awesome customization would be more adding a few links. The meat would be still a blog and a far worse one at that!

So I am back to square one but I probably should stayed here from the start anyway.

### Setting it up

[DigitalOcean](https://www.digitalocean.com) is where I hosted it to start off and being the fantastic platform that it is there's no reason to change that.

DigitalOcean also provides a 1-click setup for Ghost. This means they provide an image with Ghost already setup, no further installations steps required. Now while this is awesome I wanted something a bit less of a commitment. This is where [Docker](https://www.docker.com/) comes in.

There's already a maintained image for Ghost, `dockerfile/ghost`, so we can just pull that down with `docker pull dockerfile/ghost`.

We can then just run the container and BOOM we're done.

```sh
$ docker run -d -p 80:2368 -v <path to your content>:/ghost-override dockerfile/ghost
```

A few things are going on here.

1. This runs our blog in a production setting so if you are overriding the directory (which you should) then you need to change the production server host to `0.0.0.0`. The line of interest can be seen [here](https://github.com/TryGhost/Ghost/blob/master/config.example.js#L25).

2. The container exposes port 2368 so here we're delegating traffic on port 80 of our droplet to the port our Ghost blog is running on.

3. -v flag means we're attaching a volume to the container. In this case it's the directory of our ghost blog containing our `config.js` file and `content` directory. We're override the default directory inside the container. We pretty much have to do this otherwise all our changes will be written to a directory inside the container we have no access to. (no backups!)

So this a pretty smooth sail to get up and running. But it can get even easy easier thanks to docker-machine!

### Huh?

Very recently docker announced three new extensions to making even more awesome. [Machine](https://docs.docker.com/machine/), [swarm](https://docs.docker.com/swarm/) and [compose](https://docs.docker.com/compose/). The one we care about here is machine.

Docker machine lets us worry less about boring, mundane tasks like creating VMs, setting up ssh keys/connections and keeping track of everything. It takes care of these thigns for us! Let's work through on example in the content of the Ghost blog.

First we install docker-machine in our local machine. Click on the machine link above for downloads. Ok let's get started.

**Create the droplet**

```sh
$ docker-machine create \
    --driver digitalocean \
    --digitalocean-access-token <YOUR TOKEN> \
    blog
```

You can grab a token [here](https://cloud.digitalocean.com/settings/applications) under personal access tokens.

This creates the droplet and sets up all the ssh keys and things.

**Set the docker environment**

```sh
$ $(docker-machine env blog)
```

This allows us to type docker commands as if were working on our own local machine. For example `docker ps` will execute this command on the droplet we just spun up.

**Download the ghost image**

```sh
$ docker pull dockerfile/ghost
```

Alright cool we now have the image on our droplet. At this point we should ssh in and download our content directory for our blog. I'm hosting mine on Github but all that matters if that you're backing it up somewhere.

```sh
$ docker-machine ssh blog
```

Get your content directory on here via wget, curl, git or whatever else.

Here I put mine in the `$HOME` directory and called it `blog`

```sh
$ docker run -d -p 80:2368 -v $HOME/blog:/ghost-override dockerfile/ghost
```

BOOM!

I previously assumed that we already had ssh keys and all that set up. As you can see with the ease docker-machine provides it's not a far fetched assumption.

### Closing thoughts

We can use the workflow on other projects on the same machine. If we used the 1-click solution this would be trickier.

I only went over machine here but swarm and compose might be cool to play around with in this situation. 

We could maybe containerize our content directory and then use compose to compose it all together (bad pun).

Maybe one server isn't enough since we've become crazy popular! We could use swarm to run multiple copies of our blog on multiple servers using nginx and/or haproxy to load balance.

In conclusion docker machine is awesome. Go use it for the all dockery things!
