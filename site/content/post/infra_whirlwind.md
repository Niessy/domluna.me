+++
date = "2015-09-12T21:06:46-04:00"
title = "Whirlwind Tour of Modern Infra Tools"
tags = ["Docker", "Infra"]
+++

We're in a golden age of infrastructure tooling. Here my goal is to
give a very quick rundown of what these tools and a tiny bit about
what they do.

### Docker Engine

Docker, if you haven't heard of it (seriously?) is a container engine.
Docker containers wrap your software in an environment that contains
everything you need to run it and little more. They have similar
resource isolation and allocation benefits as VMs.

An important distinction between VMs and containers is VMs have
an entire OS associated with them, containers have the binaries
and libraries required for your app and share the kernel with other
containers. They run as isolated processes in userspace.

Here's what I think the biggest takeaways with Docker are:

1. Reproducibility - it should run on any infrastructure

Once you have docker running getting redis is as simple as `docker run redis`.
If you don't have the redis container locally it will download it, otherwise
the container with redis installed in under a second.

2. Opens the door to microservice architecture.

Because containers are isolated, reproducible and lightweight, we can think about easily
splitting our app into composable pieces/services. Say we have a Rails app
that's using Postgres for our DB and Nginx for load balancing.

We deploy a container for each service, aside from container communication we don't
have to worry much more about it. We also have the benefit here that if something
goes wrong restarting a container takes well under a second. Isolation also makes
it easier to monitor individual components (hook into individual containers).

### Docker Machine

Setting up Docker takes a bit of work; there's a fair bit of configuration involved. Docker Machine
literally does all this work for you so you can get back to running `docker` commands ASAP.

```
docker-machine create dev
```

The above command creates VM (via Virtualbox), installs/configures Docker and enables a communication
between the VM and your host so you can run `docker` commands from your host and they run on the VM.
Nifty huh?

You can change where the VM is created through the `--driver` flag. Setting this to a cloud provider
will setup to the above setup on a machine somewhere in the mystical cloud. The default is `--driver=virtualbox`.

### Docker Compose

Compose attempts to simplify linking containers and bundling them as a group. This is done through
a yaml file, typically named docker-compose.yml (name doesn't matter). So our Rails app from above
would look something like this:

```
frontend:
    image: nginx
    links:
        - app:app
app:
    build: . # Use the Dockerfile in current directory
    command: bundle exec rails s -p 3000 -b '0.0.0.0' # execute on start
    volumes:
        - .:/code
    ports:
        - "3000:3000"
    links:
        - db
db:
    image: postgres

```

The command `docker-compose up` would start up all 3 containers and link them together and mount
volumes. In this case we're telling compose to mount the code folder to the app container. Can't run
a Rails app without code!

In a production setting we would be setting up these containers on separate machines. Your
database and/or load balancer shouldn't be competing for resources.

### Docker Swarm

Swarms attacks orchestrating containers in a cluster, that is, figuring out where to put what
container where in a cluster of machines. It tries not have an opinion and allows
the you to plug and play with service discovery and scheduling solutions. I think the goal here
is to be able to pick between Kubernetes, Mesos or whatever and for it to just work.


### Etcd and Consul

[Raft](https://raft.github.io/) tackles the problem known as distributed consensus. In a nutshell this means
getting machines to agree on what has happened and what order it happened in. Before
Raft there was [Paxos](https://en.wikipedia.org/wiki/Paxos_\(computer_science\)). Nobody really had a clue what it did let alone how to implement
it. So now people just use Raft and happiness levels are up!

Etcd (pronounced "et-c-d") and Consul are built on Raft and do things like:

* Service discovery
* Health checking
* K/V store
* ...

### Kubernetes

Kubernetes figures out where to put containers in a cluster.
It's based on tech developed inside Google for several years. So yeah, it's
very likely to be good. It uses Etcd under the hood for service discovery and health
checking.

You may think the atomic unit of Kubernetes is a container but it's actually
something called a pod. A pod is a group of containers that work together. This
mirrors how containers are actually used, so it makes sense.

### Terraform

Infrastructure as code.

Here's a sample:

```
resource "aws_elb" "frontend" {
    name = "frontend-load-balancer"
    listener {
        instance_port = 8000
        instance_protocol = "http"
        lb_port = 80
        lb_protocol = "http"
    }

    instances = ["${aws_instance.app.*.id}"]
}

resource "aws_instance" "app" {
    count = 5

    ami = "ami-043a5034"
    instance_type = "m1.small"
}
```

Here we're using Elastic Load Balancer to load balance 5 EC2 m1.small instances.
Through the `terraform` tool we can then create, update and destroy our infrastructure.

So, for example, if we change count to 6 in our "aws_instance" "app" resource, Terraform will do the smart
thing and just add 1 new instance and keep the 5 already running.

A [recent post](https://segment.com/blog/the-totally-managed-analytics-pipeline/) by Segment showed a workflow using Terraform, AWS Lambda and DynamoDB.

### Prometheus

Metrics and monitoring.

Prometheus can serve as a time series database, this allows you to compute various
metrics about your system.

It can also be used to monitor your system and alert you if something goes wrong.

Prometheus also uses Etcd under the hood!

Here's an [awesome use case](http://prometheus.io/blog/2015/06/24/monitoring-dreamhack/) for Prometheus.

### Next Steps

I really didn't do any of these tools justice but I hope this gives an idea of what
they tools might be useful for and how they could help you.

Perhaps the best bit about all these tools is they're all open source. So go forth
and explore them to your hearts content!
