# source: cuda11.8.0-ubuntu22.04-oneclick/scripts/textgen-on-workspace.sh

mkdir -p /workspace

if [[ ! -d /workspace/fine-tune-experiments/ ]]; then
	# If we don't already have /workspace/fine-tune-experiments, move it there
	mv /root/fine-tune-experiments /workspace
else
	# otherwise delete the default fine-tune-experiments folder which is always re-created on pod start from the Docker
	rm -rf /root/fine-tune-experiments
fi

if [[ $PUBLIC_KEY ]]; then
	mkdir -p ~/.ssh
	chmod 700 ~/.ssh
	cd ~/.ssh
	echo "$PUBLIC_KEY" >>authorized_keys
    chmod authorized_keys
	chmod 700 -R ~/.ssh
	service ssh start
fi

sleep infinity
